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


@dataclass
class BlurStyle:
    """How a region is obscured.

    ``method`` is one of :data:`BLUR_METHODS`. ``radius`` is the kernel
    radius in pixels for ``box``/``gaussian``, ``block`` the mosaic tile
    size for ``pixelate``, and ``color`` the ``(B, G, R)`` colour for
    ``fill``. ``passes`` is how many box blurs approximate the gaussian;
    three is the usual choice and their radii are scaled so the result
    matches a single box blur of ``radius`` in variance.
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
        self.color = tuple(int(c) for c in self.color)
        if len(self.color) != 3:
            raise ValueError(
                "blur colour must be a (B, G, R) tuple, got {!r}".format(
                    self.color
                )
            )
        if not all(0 <= c <= 255 for c in self.color):
            raise ValueError(
                "blur colour values must be in 0..255, got {!r}".format(
                    self.color
                )
            )


def _scratch_get(scratch, name, shape, dtype=numpy.float32):
    """Return a work array of ``shape``, reused across calls if possible.

    ``scratch`` is a caller-owned dict (see :func:`blur_regions`); when
    it is ``None`` every call allocates its own buffer. Reuse is what
    lets a recorder blurring a fixed region settle into a steady state
    with no per-frame allocation.
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
        numpy.take(cum, hi, axis=axis, out=gather)
        numpy.take(cum, lo, axis=axis, out=work)
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

    # Leading edge — window clamped at 0, so it holds radius+1 .. 2*radius.
    head_counts = numpy.arange(
        radius + 1, 2 * radius + 1, dtype=numpy.float32
    )
    numpy.divide(
        cum[_sel(axis, slice(radius + 1, k))],
        _bcast(head_counts, axis),
        out=plane[_sel(axis, slice(0, radius))],
    )

    # Trailing edge — window clamped at n, shrinking back to radius+1.
    tail = plane[_sel(axis, slice(n - radius, n))]
    tail_counts = numpy.arange(
        2 * radius, radius, -1, dtype=numpy.float32
    )
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


def _pass_radii(radius, passes):
    """Per-pass radii whose combined variance matches one box of ``radius``.

    A box of radius ``r`` has variance ``((2r+1)**2 - 1) / 12``; variances
    add across independent passes, so each of ``passes`` passes gets
    ``1/passes`` of the target. Radii are floored at 1 — a zero-radius
    pass would be a no-op and quietly weaken the blur.
    """
    target = ((2 * radius + 1) ** 2 - 1) / float(passes)
    r = int(round((math.sqrt(target + 1.0) - 1.0) / 2.0))
    return [max(1, r)] * passes


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


def _pixelate_sub(sub, block):
    """Replace each ``block`` x ``block`` tile with its mean, in place.

    ``numpy.add.reduceat`` sums ragged runs, so a region whose size is
    not a multiple of ``block`` gets a smaller tile at the right/bottom
    edge instead of an error or a dropped strip.
    """
    h, w = sub.shape[:2]
    starts_y = numpy.arange(0, h, block)
    starts_x = numpy.arange(0, w, block)
    lens_y = numpy.diff(numpy.append(starts_y, h))
    lens_x = numpy.diff(numpy.append(starts_x, w))

    acc = numpy.add.reduceat(
        sub[..., :3].astype(numpy.float32), starts_y, axis=0
    )
    acc = numpy.add.reduceat(acc, starts_x, axis=1)
    acc /= (lens_y[:, None] * lens_x[None, :]).astype(numpy.float32)[..., None]
    acc += 0.5
    tiles = numpy.repeat(numpy.repeat(acc, lens_y, axis=0), lens_x, axis=1)
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
        reused capture buffer: a recorder blurring a fixed region ends up
        allocating nothing per frame.
    :return: ``img``.

    Each region is blurred using only the pixels inside it, so redacted
    content cannot smear outward and surrounding content cannot smear in.
    Pixels outside the regions are left byte-identical.
    """
    if style is None:
        style = BlurStyle()
    if regions is None:
        regions = [(0, 0, img.shape[1], img.shape[0])]
        origin = (0, 0)

    # Nothing to do — bail before touching the frame at all.
    if style.method in ("box", "gaussian") and style.radius <= 0:
        return img
    if style.method == "pixelate" and style.block <= 1:
        return img

    for region in regions:
        box = _clip(region, img.shape, origin)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        sub = img[y0:y1, x0:x1]
        if style.method == "fill":
            sub[..., 0:3] = style.color
        elif style.method == "pixelate":
            _pixelate_sub(sub, style.block)
        else:
            _blur_sub(sub, style, scratch)
    return img
