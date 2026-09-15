"""In-place blurring and redaction of captured BGRA frames.

Pure numpy — no Pillow, no OpenCV, no scipy — so this ships in the
default wheel and adds nothing to fastgrab's runtime dependencies. It is
also deliberately not part of the C extension: keeping it in Python means
it cannot break ``pip install`` on a distro with a missing header.

The entry point is :func:`blur_regions`, which mutates the frame handed
to it and returns it::

    from fastgrab import screenshot
    from fastgrab.effects import BlurStyle, blur_regions

    img = screenshot.Screenshot().capture()
    blur_regions(img, [(100, 100, 400, 200)], BlurStyle(method="gaussian"))

Four modes, selected by :attr:`BlurStyle.method`:

* ``box``      — a moving average; cheap and O(1) per pixel in the radius.
* ``gaussian`` — ``passes`` box blurs of matched variance; smoother.
* ``pixelate`` — block means, the classic mosaic look.
* ``pixelate-random`` — every tile a random colour.
* ``pixelate-random-shuffle`` — the real tile colours, positions permuted.
* ``fill``     — a solid ``(B, G, R)`` box, black by default.
* ``image``    — a picture stamped over the region, mapped by
  :data:`IMAGE_FITS` (``crop``, ``fit``, ``stretch``, ``tile``).

How much each one actually destroys, strongest first:

* ``fill``, ``pixelate-random`` and ``image`` — the output does not depend on the
  region's content at all, so nothing of it survives. ``fill`` says so
  plainly; ``pixelate-random`` reads as a mosaic while being just as
  final.
* ``pixelate-random-shuffle`` — the tiles keep their real colours and
  only their positions are destroyed, so the region still looks like it
  belongs. Its colour histogram survives, which leaks roughly "how much
  of what" was there.
* ``pixelate`` — layout and colour both survive at tile resolution. Text
  can be partially recovered by matching candidate renderings against the
  known tile grid.
* ``box`` and ``gaussian`` — weakest; a low radius is recoverable.

So for passwords, tokens and anything that must not leak, use ``fill``,
``pixelate-random`` or ``image``. The other three are cosmetic.

The two random modes are seeded (:attr:`BlurStyle.seed`) and therefore
identical on every frame. That is deliberate: re-rolling per frame would
let anyone average a recording back towards the plain mosaic underneath.

Conventions shared with :mod:`fastgrab.recording.clicks`: frames are
BGRA uint8, colours are ``(B, G, R)`` tuples, and ``origin`` shifts
screen-absolute coordinates into frame-local ones. Only channels 0..2 are
touched — the alpha channel is left exactly as the backend wrote it.
"""
import math
from dataclasses import dataclass, field

import numpy


BLUR_METHODS = (
    "box", "gaussian", "pixelate", "pixelate-random",
    "pixelate-random-shuffle", "fill", "image",
)

# The mosaic family — everything that reads BlurStyle.block.
PIXELATE_METHODS = ("pixelate", "pixelate-random", "pixelate-random-shuffle")

# How an image cover is mapped onto a region it does not match in shape.
IMAGE_FITS = ("crop", "fit", "stretch", "tile")

DEFAULT_RADIUS = 12
DEFAULT_BLOCK = 16
DEFAULT_PASSES = 3
DEFAULT_FILL_COLOR = (0, 0, 0)  # B, G, R — a black box
DEFAULT_SEED = 0
DEFAULT_IMAGE_FIT = "crop"

# Upper bound on how many shapes a caller-supplied scratch dict keeps
# before it is dropped, so a caller that blurs a different-sized region
# every frame can't grow it without limit.
_SCRATCH_LIMIT = 24


@dataclass(frozen=True)
class BlurStyle:
    """How a region is obscured.

    ``method`` is one of :data:`BLUR_METHODS`. ``radius`` is the kernel
    radius in pixels for ``box``/``gaussian``, ``block`` the mosaic tile
    size for ``pixelate``, and ``color`` the ``(B, G, R)`` colour for
    ``fill``, and ``image`` is the picture for ``image``, mapped onto the
    region according to ``image_fit``: ``crop`` (default) scales it to
    cover the region and trims the overflow, ``fit`` scales it to sit
    entirely inside and pads the remainder with ``color``, ``stretch``
    distorts it to the exact shape, and ``tile`` repeats it at its own
    size. Only ``stretch`` changes the picture's proportions.
    ``seed`` fixes the randomness of the ``pixelate-random``
    modes. ``passes`` is how many box blurs approximate the gaussian;
    three is the usual choice. Their radii are scaled so the combined
    variance approximates a single box blur of ``radius`` — integer radii
    can't hit it exactly, so the result may land up to 25% either side of
    it, and at small radii the pass count is reduced rather than let the
    blur come out several times stronger. See :func:`_pass_radii`.

    The dataclass is frozen. Validation happens once at construction, and
    a redaction style that could be weakened afterwards — ``style.radius
    = 0`` turning a blur into a no-op — would defeat the point of
    validating it at all.
    """

    method: str = "box"
    radius: int = DEFAULT_RADIUS
    block: int = DEFAULT_BLOCK
    passes: int = DEFAULT_PASSES
    color: tuple = DEFAULT_FILL_COLOR
    seed: int = DEFAULT_SEED
    # Excluded from __eq__/__hash__: comparing ndarrays returns an array,
    # so a generated __eq__ touching this would raise instead of answer.
    image: object = field(default=None, compare=False)
    image_fit: str = DEFAULT_IMAGE_FIT

    def __post_init__(self):
        for name in ("radius", "block", "passes", "seed"):
            value = getattr(self, name)
            try:
                whole = int(value)
            except (TypeError, ValueError):
                # int(nan) and int("x") land here, so NaN cannot slip past
                # the range checks below by failing every comparison.
                raise ValueError(
                    "blur {} must be a whole number of pixels, got "
                    "{!r}".format(name, value)
                )
            if whole != value:
                raise ValueError(
                    "blur {} must be a whole number of pixels, got "
                    "{!r}".format(name, value)
                )
            object.__setattr__(self, name, whole)
        if self.method not in BLUR_METHODS:
            raise ValueError(
                "unknown blur method {!r}; expected one of {}".format(
                    self.method, ", ".join(BLUR_METHODS)
                )
            )
        if self.radius < 0:
            raise ValueError(
                "blur radius must be non-negative, got {}".format(self.radius)
            )
        if self.block < 1:
            raise ValueError(
                "blur block size must be at least 1, got {}".format(self.block)
            )
        if self.passes < 1:
            raise ValueError(
                "blur passes must be at least 1, got {}".format(self.passes)
            )
        if self.seed < 0:
            raise ValueError(
                "blur seed must not be negative, got {}".format(self.seed)
            )
        # Reject settings that are silently identity operations. This is a
        # redaction API: a style that quietly leaves the pixels readable is
        # the one failure mode that actually leaks, so it has to be an
        # error rather than a no-op. Values irrelevant to the chosen
        # method are left alone.
        if self.method in ("box", "gaussian") and self.radius < 1:
            raise ValueError(
                "a {} blur of radius {} leaves the region unchanged; use 1 "
                "or more".format(self.method, self.radius)
            )
        if self.method in PIXELATE_METHODS and self.block < 2:
            raise ValueError(
                "a {} block of {} leaves the region unchanged; use 2 "
                "or more".format(self.method, self.block)
            )
        try:
            color = tuple(self.color)
        except TypeError:
            raise ValueError(
                "blur colour must be a (B, G, R) tuple, got {!r}".format(
                    self.color
                )
            )
        if len(color) != 3:
            raise ValueError(
                "blur colour must be a (B, G, R) tuple, got {!r}".format(
                    self.color
                )
            )
        try:
            color = tuple(int(c) for c in color)
        except (TypeError, ValueError):
            raise ValueError(
                "blur colour values must be integers, got {!r}".format(
                    self.color
                )
            )
        if not all(0 <= c <= 255 for c in color):
            raise ValueError(
                "blur colour values must be in 0..255, got {!r}".format(
                    self.color
                )
            )
        # Frozen, so normalising the colour needs the back door.
        object.__setattr__(self, "color", color)

        if self.image_fit not in IMAGE_FITS:
            raise ValueError(
                "unknown image fit {!r}; expected one of {}".format(
                    self.image_fit, ", ".join(IMAGE_FITS)
                )
            )
        if self.method == "image" and self.image is None:
            raise ValueError(
                "blur method 'image' needs image= set to a (H, W, 3) or "
                "(H, W, 4) uint8 array"
            )
        if self.image is not None:
            object.__setattr__(self, "image", _as_cover_image(self.image))


def _as_cover_image(image):
    """Validate a cover image and take an immutable BGR snapshot of it.

    Copied rather than referenced, and marked read-only: a caller who
    mutated the array afterwards would silently change what every later
    capture paints over the secret.
    """
    array = numpy.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(
            "blur image must be (H, W, 3) or (H, W, 4); got shape "
            "{!r}".format(getattr(array, "shape", None))
        )
    if array.dtype != numpy.uint8:
        raise ValueError(
            "blur image must be uint8, like the frames it covers; got "
            "{}".format(array.dtype)
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("blur image must have at least one pixel")
    # numpy.ascontiguousarray returns the *same* object when the input is
    # already contiguous, so it is not a snapshot — and marking that
    # read-only would freeze the caller's own array. Force the copy.
    snapshot = numpy.array(array[..., :3], dtype=numpy.uint8, order="C",
                           copy=True)
    snapshot.flags.writeable = False
    return snapshot


def _axis_samples(count, start, span, limit):
    """Nearest-neighbour source indices for ``count`` output positions.

    Sampled at pixel centres over ``[start, start + span)`` of the source
    axis, then clamped — off-by-one at the last row is the classic way a
    resize picks up a stripe of whatever follows the image in memory.
    """
    taps = start + (numpy.arange(count) + 0.5) * (span / float(count))
    return numpy.clip(taps.astype(numpy.int64), 0, limit - 1)


def _cover_index(scratch, src_h, src_w, dst_h, dst_w, fit):
    """Row/column source indices and the offset to draw them at.

    Returns ``(rows, cols, y_off, x_off)``. ``len(rows)`` and
    ``len(cols)`` are the drawn size, which equals the region for every
    fit except ``fit``, where the image is inset and the caller pads
    around it.
    """
    key = ("cover", src_h, src_w, dst_h, dst_w, fit)
    if scratch is not None:
        cached = scratch.get(key)
        if cached is not None:
            return cached

    if fit == "stretch":
        out = (_axis_samples(dst_h, 0.0, src_h, src_h),
               _axis_samples(dst_w, 0.0, src_w, src_w), 0, 0)
    elif fit == "tile":
        out = (numpy.arange(dst_h) % src_h, numpy.arange(dst_w) % src_w, 0, 0)
    elif fit == "crop":
        # Cover: the larger scale wins, so the region is filled and the
        # overflowing axis is trimmed evenly from both sides.
        scale = max(dst_h / float(src_h), dst_w / float(src_w))
        win_h = min(float(src_h), dst_h / scale)
        win_w = min(float(src_w), dst_w / scale)
        out = (_axis_samples(dst_h, (src_h - win_h) / 2.0, win_h, src_h),
               _axis_samples(dst_w, (src_w - win_w) / 2.0, win_w, src_w),
               0, 0)
    else:  # "fit" — contain: the smaller scale wins, nothing is cut off
        scale = min(dst_h / float(src_h), dst_w / float(src_w))
        drawn_h = max(1, min(dst_h, int(round(src_h * scale))))
        drawn_w = max(1, min(dst_w, int(round(src_w * scale))))
        out = (_axis_samples(drawn_h, 0.0, src_h, src_h),
               _axis_samples(drawn_w, 0.0, src_w, src_w),
               (dst_h - drawn_h) // 2, (dst_w - drawn_w) // 2)

    if scratch is not None:
        scratch[key] = out
    return out


def _image_sub(sub, style, scratch):
    """Stamp ``style.image`` over the region in place.

    Nearest neighbour on purpose: the cover is there to hide what is
    underneath, not to look smooth, and it keeps this pure numpy with no
    interpolation pass over the frame.
    """
    h, w = sub.shape[:2]
    image = style.image
    src_h, src_w = image.shape[:2]
    rows, cols, y_off, x_off = _cover_index(
        scratch, src_h, src_w, h, w, style.image_fit
    )
    drawn_h, drawn_w = len(rows), len(cols)

    if drawn_h != h or drawn_w != w:
        # "fit" leaves margins; paint them before the picture goes down so
        # no part of the region is left showing what it was meant to hide.
        sub[..., 0:3] = style.color

    rows_buf = _scratch_get(
        scratch, "cover_rows", (drawn_h, src_w, 3), numpy.uint8
    )
    numpy.take(image, rows, axis=0, out=rows_buf, mode="clip")
    cover = _scratch_get(scratch, "cover", (drawn_h, drawn_w, 3), numpy.uint8)
    numpy.take(rows_buf, cols, axis=1, out=cover, mode="clip")
    target = sub[y_off:y_off + drawn_h, x_off:x_off + drawn_w]
    numpy.copyto(target[..., :3], cover)


def _normalise_regions(regions):
    """Validate an iterable of rectangles and materialise it into a tuple.

    Shared by :func:`blur_regions` and
    :func:`fastgrab.screenshot._normalise_blur` so the low-level entry
    point is no less strict than the capture API.

    Materialising matters because a caller can reasonably pass a
    generator, and a stored one-shot iterable would redact one frame and
    silently leave every later one in the clear. Empty and fractional
    rectangles raise rather than being skipped: an empty rectangle looks
    like a real target to the caller but covers nothing, and truncating a
    fractional origin can shift the box off a sliver of what it was meant
    to hide. Rectangles that merely fall outside the frame are a
    different matter — those are clipped away silently by :func:`_clip`,
    because a screen-absolute region legitimately misses a sub-region
    capture.
    """
    # Checked before consuming: an exhausted generator and an empty one
    # are indistinguishable afterwards.
    is_iterator = hasattr(regions, "__next__")
    out = []
    for region in regions:
        values = tuple(region)
        if len(values) != 4:
            raise ValueError(
                "blur regions must be (x, y, width, height); got "
                "{!r}".format(region)
            )
        whole = []
        for value in values:
            as_int = int(value)
            if as_int != value:
                raise ValueError(
                    "blur region coordinates must be whole pixels; got {!r}"
                    " — round them yourself so the rectangle lands where "
                    "you meant".format(region)
                )
            whole.append(as_int)
        if whole[2] <= 0 or whole[3] <= 0:
            raise ValueError(
                "blur region width and height must be positive; got "
                "{!r}".format(region)
            )
        out.append(tuple(whole))
    if not out and is_iterator:
        # Reusing one generator across two calls would redact the first
        # frame and quietly leave the rest in the clear. An empty list is
        # still an honest "do nothing".
        raise ValueError(
            "blur regions iterable was empty or already consumed; pass "
            "[] to do nothing, and hold a list rather than a generator "
            "to reuse regions across calls"
        )
    return tuple(out)


def _scratch_get(scratch, name, shape, dtype=numpy.float32):
    """Return a work array of ``shape``, reused across calls if possible.

    ``scratch`` is a caller-owned dict (see :func:`blur_regions`); when
    it is ``None`` every call allocates its own buffer. Reuse is what
    lets a recorder blurring a fixed region stop allocating work buffers
    per frame. It does not reach zero bytes: numpy keeps a fixed ~100 KiB
    iteration buffer for ufuncs whose output is a strided view (the
    column slices in :func:`_moving_average`), which is constant no
    matter how large the region is.
    """
    if scratch is None:
        return numpy.empty(shape, dtype)
    if len(scratch) > _SCRATCH_LIMIT:
        scratch.clear()
    key = (name, shape, numpy.dtype(dtype).str)
    buf = scratch.get(key)
    if buf is None:
        buf = numpy.empty(shape, dtype)
        scratch[key] = buf
    return buf


def _window_index(scratch, length, radius):
    """Half-open ``[lo, hi)`` window bounds and widths for a 1-D axis.

    The window is clamped to the axis, so pixels near an edge average
    over fewer neighbours instead of over replicated padding — the two
    are visually equivalent and this needs no padded copy of the frame.
    ``hi``/``lo`` index into a cumulative sum with a leading zero, hence
    the ``length + 1`` valid range.
    """
    key = ("win", length, radius)
    if scratch is not None:
        cached = scratch.get(key)
        if cached is not None:
            return cached
    idx = numpy.arange(length)
    lo = numpy.maximum(idx - radius, 0)
    hi = numpy.minimum(idx + radius + 1, length)
    counts = (hi - lo).astype(numpy.float32)
    out = (lo, hi, counts)
    if scratch is not None:
        scratch[key] = out
    return out


def _edge_counts(scratch, radius):
    """Per-position window widths for the clamped head and tail positions.

    Cached like :func:`_window_index`: these are rebuilt once per axis,
    per pass, per channel otherwise, which is pure waste in a capture
    loop even though each array is only ``radius`` long.
    """
    key = ("edge", radius)
    if scratch is not None:
        cached = scratch.get(key)
        if cached is not None:
            return cached
    head = numpy.arange(radius + 1, 2 * radius + 1, dtype=numpy.float32)
    tail = numpy.arange(2 * radius, radius, -1, dtype=numpy.float32)
    out = (head, tail)
    if scratch is not None:
        scratch[key] = out
    return out


def _sel(axis, s):
    """Index tuple selecting ``s`` along ``axis`` of a 2-D array."""
    return (slice(None), s) if axis == 1 else (s, slice(None))


def _bcast(counts, axis):
    """Shape a per-position divisor so it broadcasts along ``axis``."""
    return counts if axis == 1 else counts.reshape(-1, 1)


def _moving_average(plane, radius, axis, scratch):
    """Average each pixel over ``[-radius, +radius]`` along ``axis``, in place.

    Computed from a cumulative sum, so the cost per pixel is independent
    of ``radius`` — a 40 px blur costs the same as a 4 px one, where a
    direct convolution would be O(radius**2) per pixel.

    Away from the edges both window bounds are in range, which makes the
    two sums plain contiguous slices of the cumulative sum. That is the
    hot path and it does no fancy indexing at all; only the ``radius``
    positions at each end need per-position divisors, because their
    windows are clamped to the axis and so average over fewer pixels.
    """
    n = plane.shape[axis]
    k = 2 * radius + 1

    shape = list(plane.shape)
    shape[axis] += 1
    cum = _scratch_get(scratch, "cum{}".format(axis), tuple(shape))
    cum[_sel(axis, slice(0, 1))] = 0
    numpy.cumsum(plane, axis=axis, out=cum[_sel(axis, slice(1, None))])

    if 2 * radius >= n:
        # Kernel wider than the axis: every window is clamped, so there
        # is no contiguous interior and each position needs its own
        # bounds. Rare (a tiny region with a big radius) and not worth
        # special-casing beyond falling back to a gather.
        lo, hi, counts = _window_index(scratch, n, radius)
        work = _scratch_get(scratch, "work", plane.shape)
        gather = _scratch_get(scratch, "gather", plane.shape)
        # mode="clip" never clips here — lo/hi are built from this
        # axis's own length — but the default mode="raise" makes
        # numpy.take allocate a full-size internal temporary even when
        # out= is given, which defeats the scratch buffers entirely.
        numpy.take(cum, hi, axis=axis, out=gather, mode="clip")
        numpy.take(cum, lo, axis=axis, out=work, mode="clip")
        numpy.subtract(gather, work, out=plane)
        numpy.divide(plane, _bcast(counts, axis), out=plane)
        return

    # Interior — the full kernel fits, so every window has k pixels.
    body = plane[_sel(axis, slice(radius, n - radius))]
    numpy.subtract(
        cum[_sel(axis, slice(k, n + 1))],
        cum[_sel(axis, slice(0, n + 1 - k))],
        out=body,
    )
    numpy.divide(body, k, out=body)

    head_counts, tail_counts = _edge_counts(scratch, radius)

    # Leading edge — window clamped at 0, so it holds radius+1 .. 2*radius.
    numpy.divide(
        cum[_sel(axis, slice(radius + 1, k))],
        _bcast(head_counts, axis),
        out=plane[_sel(axis, slice(0, radius))],
    )

    # Trailing edge — window clamped at n, shrinking back to radius+1.
    tail = plane[_sel(axis, slice(n - radius, n))]
    numpy.subtract(
        cum[_sel(axis, slice(n, n + 1))],
        cum[_sel(axis, slice(n - 2 * radius, n - radius))],
        out=tail,
    )
    numpy.divide(tail, _bcast(tail_counts, axis), out=tail)


def _box_pass(plane, radius, scratch):
    """One separable box blur of ``plane`` (float32, 2-D), in place."""
    _moving_average(plane, radius, 1, scratch)
    _moving_average(plane, radius, 0, scratch)


# How far the combined variance of the box passes may drift from the
# variance of the single box blur the caller asked for. Integer radii
# can't hit it exactly; a quarter is loose enough to keep three passes at
# small radii and tight enough to reject a visibly wrong strength.
_VARIANCE_TOLERANCE = 0.25


def _pass_radii(radius, passes):
    """Per-pass radii approximating one box blur of ``radius``.

    A box of radius ``r`` has variance ``((2r+1)**2 - 1) / 12``, and
    variances add across independent passes, so each of ``passes`` passes
    should carry ``1/passes`` of the target. Integer radii can only
    approximate that, so this walks the pass count down from ``passes``
    and takes the first one whose combined variance lands within
    :data:`_VARIANCE_TOLERANCE` of the target.

    The tolerance is symmetric: the result can be up to a quarter weaker
    or stronger than the requested radius, whichever integer radius comes
    closest. Both neighbours of the ideal real radius are tried, because
    rounding to the nearer one is not the same as picking the one whose
    variance is nearer — at radius 19 over 24 passes, rounding picks a
    radius that misses the envelope while its floor sits inside it.

    Dropping passes matters at small radii: the ideal share can round to
    a zero radius, and flooring it at 1 instead — as an earlier version
    did — makes the blur far stronger than asked. Three passes of radius
    1 have variance 2.0 where a single radius-1 box has 0.67, i.e. 3x too
    strong. Fewer, honest passes beat a silently wrong strength.
    """
    target = ((2 * radius + 1) ** 2 - 1) / 12.0
    for n in range(passes, 0, -1):
        ideal = (math.sqrt(12.0 * (target / n) + 1.0) - 1.0) / 2.0
        candidates = {
            max(1, int(math.floor(ideal))), max(1, int(math.ceil(ideal))),
        }
        error, radii = min(
            (abs(n * ((2 * r + 1) ** 2 - 1) / 12.0 - target), [r] * n)
            for r in candidates
        )
        if error <= _VARIANCE_TOLERANCE * target:
            return radii
    # n == 1 reproduces the requested radius exactly, so this is only
    # reached if radius itself is degenerate — which BlurStyle rejects.
    return [max(1, radius)]


def _blur_sub(sub, style, scratch):
    """Apply ``box``/``gaussian`` to the BGR channels of ``sub`` in place."""
    h, w = sub.shape[:2]
    radii = (
        [style.radius] if style.method == "box"
        else _pass_radii(style.radius, style.passes)
    )
    # One channel at a time: the float32 work arrays are then h*w*4 bytes
    # rather than three times that, which matters for a full-frame blur.
    plane = _scratch_get(scratch, "plane", (h, w))
    for channel in range(3):
        plane[...] = sub[..., channel]
        for radius in radii:
            _box_pass(plane, radius, scratch)
        # +0.5 so the cast rounds instead of truncating; averages of
        # 0..255 values never reach 256, so this cannot wrap.
        numpy.add(plane, 0.5, out=plane)
        numpy.copyto(sub[..., channel], plane, casting="unsafe")


def _pixelate_plan(scratch, h, w, block):
    """Tile boundaries, per-tile pixel counts and the expansion indices.

    All of it depends only on ``(h, w, block)``, so it is cached rather
    than rebuilt per frame.
    """
    key = ("pix", h, w, block)
    if scratch is not None:
        cached = scratch.get(key)
        if cached is not None:
            return cached
    starts_y = numpy.arange(0, h, block)
    starts_x = numpy.arange(0, w, block)
    lens_y = numpy.diff(numpy.append(starts_y, h))
    lens_x = numpy.diff(numpy.append(starts_x, w))
    counts = (
        (lens_y[:, None] * lens_x[None, :]).astype(numpy.float32)[..., None]
    )
    # Which tile each output row / column reads from, so the means can be
    # expanded back with numpy.take(out=...) instead of numpy.repeat,
    # which has no out= and would allocate the full tile image per frame.
    idx_y = numpy.repeat(numpy.arange(len(starts_y)), lens_y)
    idx_x = numpy.repeat(numpy.arange(len(starts_x)), lens_x)
    plan = (starts_y, starts_x, counts, idx_y, idx_x)
    if scratch is not None:
        scratch[key] = plan
    return plan


def _random_tiles(scratch, n_y, n_x, seed):
    """A fixed random colour per tile, cached.

    Drawn once per (grid, seed) and reused, so a capture loop neither
    reallocates it nor — more importantly — re-rolls it. Re-rolling every
    frame would look stronger and be weaker: averaging enough frames of a
    recording converges on whatever is underneath.
    """
    key = ("randtiles", n_y, n_x, seed)
    if scratch is not None:
        cached = scratch.get(key)
        if cached is not None:
            return cached
    rng = numpy.random.default_rng(seed)
    tiles = rng.integers(0, 256, (n_y, n_x, 3)).astype(numpy.float32)
    if scratch is not None:
        scratch[key] = tiles
    return tiles


def _tile_permutation(scratch, count, seed):
    """A fixed permutation of ``count`` tiles, cached like the colours."""
    key = ("randperm", count, seed)
    if scratch is not None:
        cached = scratch.get(key)
        if cached is not None:
            return cached
    perm = numpy.random.default_rng(seed).permutation(count)
    if scratch is not None:
        scratch[key] = perm
    return perm


def _pixelate_sub(sub, style, scratch):
    """Replace each tile with a single colour, in place.

    Which colour depends on the method: the tile's own mean
    (``pixelate``), a fixed random one (``pixelate-random``), or another
    tile's mean (``pixelate-random-shuffle``).

    ``numpy.add.reduceat`` sums ragged runs, so a region whose size is
    not a multiple of ``block`` gets a smaller tile at the right/bottom
    edge instead of an error or a dropped strip.
    """
    h, w = sub.shape[:2]
    block = style.block
    starts_y, starts_x, counts, idx_y, idx_x = _pixelate_plan(
        scratch, h, w, block
    )
    n_y, n_x = len(starts_y), len(starts_x)
    acc = _scratch_get(scratch, "pix_acc", (n_y, n_x, 3))

    if style.method == "pixelate-random":
        # Nothing is read from the frame at all, which is exactly why
        # this destroys the content as completely as fill does.
        acc[...] = _random_tiles(scratch, n_y, n_x, style.seed)
    else:
        src = _scratch_get(scratch, "pix_src", (h, w, 3))
        numpy.copyto(src, sub[..., :3])
        rows = _scratch_get(scratch, "pix_rows", (n_y, w, 3))
        numpy.add.reduceat(src, starts_y, axis=0, out=rows)
        numpy.add.reduceat(rows, starts_x, axis=1, out=acc)
        numpy.divide(acc, counts, out=acc)
        # +0.5 so the cast back to uint8 rounds instead of truncating.
        numpy.add(acc, 0.5, out=acc)

        if style.method == "pixelate-random-shuffle":
            # Real colours, scrambled positions: the region keeps its
            # palette and loses its layout.
            perm = _tile_permutation(scratch, n_y * n_x, style.seed)
            flat = acc.reshape(-1, 3)
            shuffled = _scratch_get(scratch, "pix_shuf", (n_y * n_x, 3))
            numpy.take(flat, perm, axis=0, out=shuffled, mode="clip")
            flat[...] = shuffled

    # mode="clip" as in _moving_average: the indices are in range by
    # construction, and it keeps numpy.take off the path that allocates a
    # full-size temporary despite out=.
    spread_y = _scratch_get(scratch, "pix_spread", (h, n_x, 3))
    numpy.take(acc, idx_y, axis=0, out=spread_y, mode="clip")
    tiles = _scratch_get(scratch, "pix_tiles", (h, w, 3))
    numpy.take(spread_y, idx_x, axis=1, out=tiles, mode="clip")
    numpy.copyto(sub[..., :3], tiles, casting="unsafe")


def _clip(region, shape, origin):
    """Translate ``region`` by ``-origin`` and intersect it with the frame.

    Returns ``(y0, y1, x0, x1)`` or ``None`` when nothing is left. Doing
    the clamp explicitly matters: handing a negative coordinate straight
    to a slice would wrap around and blur the wrong side of the frame.
    """
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return None
    frame_h, frame_w = shape[:2]
    x0 = max(int(x) - int(origin[0]), 0)
    y0 = max(int(y) - int(origin[1]), 0)
    x1 = min(int(x) - int(origin[0]) + int(w), frame_w)
    y1 = min(int(y) - int(origin[1]) + int(h), frame_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def blur_regions(img, regions=None, style=None, origin=(0, 0),
                 scratch=None) -> numpy.ndarray:
    """Obscure parts of a BGRA frame in place and return it.

    :param img: the frame, a ``(H, W, 4)`` uint8 array such as the one
        :meth:`fastgrab.screenshot.Screenshot.capture` returns. It is
        modified in place; only channels 0..2 are written.
    :param regions: an iterable of ``(x, y, w, h)`` rectangles, or
        ``None`` for the whole frame. Rectangles are clipped to the
        frame; ones that fall entirely outside it are skipped. The
        iterable is consumed once. A generator handed to a second call
        arrives empty, and that raises rather than quietly doing nothing
        — an exhausted iterator is indistinguishable from an empty one,
        and silently skipping the redaction is the worse reading. Hold a
        list instead, or set ``Screenshot(blur=...)`` / ``grab.blur =
        ...``, which materialise once and reuse the result for every
        capture. A per-call ``capture(blur=...)`` override stores
        nothing, so it is one-shot in the same way.
    :param style: a :class:`BlurStyle`; ``None`` means the defaults
        (a box blur of radius 12).
    :param origin: the frame's top-left corner in the caller's
        coordinate system, subtracted from every region — pass the
        capture bbox's ``(x, y)`` to give regions in screen-absolute
        coordinates, exactly like ``bbox_origin`` in
        :func:`fastgrab.recording.clicks.overlay_clicks`.
    :param scratch: an optional dict the caller keeps between calls, used
        to reuse the float32 work arrays. Same idea as ``Screenshot``'s
        reused capture buffer: a recorder blurring a fixed region stops
        allocating work buffers per frame. What remains is numpy's own
        fixed iteration buffer (~100 KiB), which does not grow with the
        region.
    :return: ``img``.

    Each region is blurred using only the pixels inside it, so redacted
    content cannot smear outward and surrounding content cannot smear in.
    Pixels outside the regions are left byte-identical.
    """
    if style is None:
        style = BlurStyle()
    elif not isinstance(style, BlurStyle):
        # Deliberately not duck-typed. Reading attributes off whatever
        # arrives, with defaults for the missing ones, meant a stray
        # value silently became a default box blur: blur_regions(img,
        # None, "fill") softened the region instead of painting it out,
        # and only fill actually destroys pixels.
        raise TypeError(
            "style must be a BlurStyle or None, got {!r}".format(
                type(style).__name__
            )
        )

    if regions is None:
        regions = ((0, 0, img.shape[1], img.shape[0]),)
        origin = (0, 0)
    else:
        regions = _normalise_regions(regions)

    for region in regions:
        box = _clip(region, img.shape, origin)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        sub = img[y0:y1, x0:x1]
        if style.method == "fill":
            sub[..., 0:3] = style.color
        elif style.method == "image":
            _image_sub(sub, style, scratch)
        elif style.method in PIXELATE_METHODS:
            _pixelate_sub(sub, style, scratch)
        else:
            _blur_sub(sub, style, scratch)  # box / gaussian
    return img
