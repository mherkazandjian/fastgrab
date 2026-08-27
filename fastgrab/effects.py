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
* ``fill``     — a solid ``(B, G, R)`` box, black by default.

**Only ``fill`` actually destroys the pixels.** ``box``, ``gaussian`` and
``pixelate`` are cosmetic: they discard high-frequency detail but a
determined attacker can partially recover text from a low-radius blur or
a coarse mosaic. Use ``fill`` for passwords, tokens and anything else
that must not leak.

Conventions shared with :mod:`fastgrab.recording.clicks`: frames are
BGRA uint8, colours are ``(B, G, R)`` tuples, and ``origin`` shifts
screen-absolute coordinates into frame-local ones. Only channels 0..2 are
touched — the alpha channel is left exactly as the backend wrote it.
"""
import math
from dataclasses import dataclass

import numpy


BLUR_METHODS = ("box", "gaussian", "pixelate", "fill")

DEFAULT_RADIUS = 12
DEFAULT_BLOCK = 16
DEFAULT_PASSES = 3
DEFAULT_FILL_COLOR = (0, 0, 0)  # B, G, R — a black box

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
    ``fill``. ``passes`` is how many box blurs approximate the gaussian;
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

    def __post_init__(self):
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
        if self.method == "pixelate" and self.block < 2:
            raise ValueError(
                "a pixelate block of {} leaves the region unchanged; use 2 "
                "or more".format(self.block)
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


def _pixelate_sub(sub, block, scratch):
    """Replace each ``block`` x ``block`` tile with its mean, in place.

    ``numpy.add.reduceat`` sums ragged runs, so a region whose size is
    not a multiple of ``block`` gets a smaller tile at the right/bottom
    edge instead of an error or a dropped strip.
    """
    h, w = sub.shape[:2]
    starts_y, starts_x, counts, idx_y, idx_x = _pixelate_plan(
        scratch, h, w, block
    )
    n_y, n_x = len(starts_y), len(starts_x)

    src = _scratch_get(scratch, "pix_src", (h, w, 3))
    numpy.copyto(src, sub[..., :3])
    rows = _scratch_get(scratch, "pix_rows", (n_y, w, 3))
    numpy.add.reduceat(src, starts_y, axis=0, out=rows)
    acc = _scratch_get(scratch, "pix_acc", (n_y, n_x, 3))
    numpy.add.reduceat(rows, starts_x, axis=1, out=acc)
    numpy.divide(acc, counts, out=acc)
    # +0.5 so the cast back to uint8 rounds instead of truncating.
    numpy.add(acc, 0.5, out=acc)

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
        frame; ones that fall entirely outside it are skipped.
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
        # Duck-typed styles are allowed, but they get the full BlurStyle
        # validation rather than a subset of it — a loose style with
        # passes=0 or a colour that is really an image would otherwise
        # return the frame untouched and look like a successful redaction.
        style = BlurStyle(
            method=getattr(style, "method", "box"),
            radius=getattr(style, "radius", DEFAULT_RADIUS),
            block=getattr(style, "block", DEFAULT_BLOCK),
            passes=getattr(style, "passes", DEFAULT_PASSES),
            color=getattr(style, "color", DEFAULT_FILL_COLOR),
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
        elif style.method == "pixelate":
            _pixelate_sub(sub, style.block, scratch)
        else:
            _blur_sub(sub, style, scratch)  # box / gaussian
    return img
