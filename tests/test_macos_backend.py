"""Tests for the macOS backend: the points-vs-pixels split on Retina.

The rect arithmetic is a pure function, tested directly. The copy path
memmoves out of a CFData pointer, so it gets a fake ctypes layer — the
only place ``off_x``/``off_y`` ever run, since CI Macs are 1x.

The fakes deliberately model CoreGraphics where it is *unforgiving*:
``_FakeCoreGraphics._image`` refuses an out-of-bounds rect rather than
letting numpy clip it into a zero-padded image, the display mode can
under-report its pixel size so an oversized capture can be exercised,
and CoreGraphics and CoreFoundation are separate objects so a symbol
called on the wrong framework handle fails here the way it would on a
real machine.

The provider is modelled as real bytes for the same reason. It can be
over-allocated (``provider_pad`` — issue #46: a 1920x1080 capture whose
CFData is padded out to the 16 KiB arm64 page), and it can be a
*parent* bitmap the image is merely a window onto (``parent_margin``),
where the pixels start at a non-zero offset and the byte pointer aims
at somebody else's top-left corner. Everything outside the image is
filled with 0xA5, so a capture read at the wrong offset or stride comes
back with wrong *values*, not merely a different call count.
"""
import ctypes
from types import SimpleNamespace

import numpy
import pytest

from fastgrab.backends.macos import (
    MacosBackend,
    _snap_rect_to_points,
    _K_CG_BITMAP_BYTE_ORDER_32_LITTLE,
)

# Everything here runs against fakes — no display server, on any platform.
pytestmark = pytest.mark.no_display

# a 40x30-pixel backing store; RETINA drives it from a 20x15-point desktop
RETINA = dict(pixel_w=40, pixel_h=30, point_w=20, point_h=15)
FLAT = dict(pixel_w=40, pixel_h=30, point_w=40, point_h=30)


@pytest.mark.parametrize("rect, expected", [
    # (x, y, w, h) -> (pt_x, pt_y, pt_w, pt_h, off_x, off_y)
    ((0, 0, 8, 6), (0, 0, 4, 3, 0, 0)),        # aligned to the scale
    ((2, 4, 10, 8), (1, 2, 5, 4, 0, 0)),       # even offset and size
    ((7, 5, 11, 9), (3, 2, 6, 5, 1, 1)),       # odd -> snaps out, offset 1,1
    ((1, 1, 1, 1), (0, 0, 1, 1, 1, 1)),        # single pixel inside a point
    ((39, 29, 1, 1), (19, 14, 1, 1, 1, 1)),    # bottom-right corner
])
def test_rect_snaps_outward_to_whole_points(rect, expected):
    assert _snap_rect_to_points(*rect, **RETINA) == expected


@pytest.mark.parametrize("rect", [
    (0, 0, 8, 6),
    (7, 5, 11, 9),
    (1, 1, 1, 1),        # degenerate single pixel, the 1x twin of the above
    (39, 29, 1, 1),
])
def test_rect_is_unchanged_without_scaling(rect):
    x, y, w, h = rect
    assert _snap_rect_to_points(*rect, **FLAT) == (x, y, w, h, 0, 0)


# Colour-space handles. Plain strings, so a mix-up shows up by name.
_DEVICE_RGB = "device-rgb"
_IMAGE_RGB = "image-rgb"

# kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little. Spelled
# out rather than imported from the backend: this is the ABI constant
# that makes the destination BGRA, and the point of asserting on it is
# to catch the backend changing it.
_BGRA8888 = 2 | (2 << 12)


class _FakeImage:
    """The layout CoreGraphics knows about; the backend can only ask.

    ``data_offset`` is where this image's first pixel sits inside the
    provider. It is non-zero exactly when the image is a window onto a
    larger parent bitmap — the case the byte pointer cannot reveal.
    """

    def __init__(self, w, h, stride, data_offset, buf):
        self.w = w
        self.h = h
        self.stride = stride
        self.data_offset = data_offset
        self.buf = buf


class _FakeCoreGraphics:
    """The slice of CoreGraphics the backend calls, over a numpy screen."""

    def __init__(self, screen, point_w, point_h, row_pad=0,
                 mode_pixel_w=None, mode_pixel_h=None, provider_pad=0,
                 parent_margin=0):
        self.screen = screen
        self.pixel_h, self.pixel_w = screen.shape[:2]
        self.point_w, self.point_h = point_w, point_h
        self.row_pad = row_pad
        # Bytes the provider holds beyond the image's own rows. Legal per
        # CGImageCreate, and what issue #46 hits.
        self.provider_pad = provider_pad
        # Pixels of surrounding parent bitmap the provider also covers.
        self.parent_margin = parent_margin
        # What CGDisplayModeGetPixel{Width,Height} claim. Defaults to the
        # real backing store; a test can make the mode under-report to
        # model a display whose mode dict omits the HiDPI pixel fields.
        self.mode_pixel_w = self.pixel_w if mode_pixel_w is None else mode_pixel_w
        self.mode_pixel_h = self.pixel_h if mode_pixel_h is None else mode_pixel_h
        self.bits_per_component = 8
        self.main_display = 1
        # The colour space CGImageGetColorSpace hands back, and the ones
        # CGBitmapContextCreate refuses (an EDR profile needs float
        # components, so a real one refuses and logs).
        self.image_colorspace = _IMAGE_RGB
        self.refuse_colorspaces = set()
        self.colorspace_queries = 0
        self.created_colorspaces = []
        self.released_colorspaces = []
        self.copy_data_calls = 0
        self.contexts = {}
        self.open_contexts = set()
        self._meta = {}
        self._buffers = {}

    def CGMainDisplayID(self):
        return self.main_display

    def CGDisplayCopyDisplayMode(self, display):
        return 1

    def CGDisplayModeRelease(self, mode):
        pass

    def CGDisplayModeGetWidth(self, mode):
        return self.point_w

    def CGDisplayModeGetHeight(self, mode):
        return self.point_h

    def CGDisplayModeGetPixelWidth(self, mode):
        return self.mode_pixel_w

    def CGDisplayModeGetPixelHeight(self, mode):
        return self.mode_pixel_h

    def _image(self, x, y, w, h):
        if x < 0 or y < 0 or x + w > self.pixel_w or y + h > self.pixel_h:
            # numpy would clip this to a short slice and leave the rest of
            # the buffer zeroed while the metadata still claimed the full
            # size — a silently "valid" image no real API would return.
            raise AssertionError(
                "rect %r falls outside the %dx%d screen; real CoreGraphics "
                "clips to the display bounds and returns a smaller image"
                % ((x, y, w, h), self.pixel_w, self.pixel_h))
        # The provider covers the parent rect, which is this image grown
        # by parent_margin and clipped to the screen. At margin 0 the two
        # coincide and the image owns its buffer outright.
        margin = self.parent_margin
        par_x = max(0, x - margin)
        par_y = max(0, y - margin)
        par_w = min(self.pixel_w, x + w + margin) - par_x
        par_h = min(self.pixel_h, y + h + margin) - par_y
        stride = par_w * 4 + self.row_pad
        buf = (ctypes.c_ubyte * (stride * par_h + self.provider_pad))()
        # Fill first, so row padding and any over-allocation hold
        # something recognisable rather than zeros: pixels fetched from
        # the wrong place then differ in value, not just in provenance.
        ctypes.memset(buf, 0xA5, len(buf))
        for index, row in enumerate(
                self.screen[par_y:par_y + par_h, par_x:par_x + par_w]):
            ctypes.memmove(ctypes.byref(buf, index * stride), row.tobytes(),
                           par_w * 4)
        token = len(self._meta) + 1
        self._meta[token] = _FakeImage(
            w, h, stride, (y - par_y) * stride + (x - par_x) * 4, buf)
        self._buffers[token] = buf
        return token

    def CGDisplayCreateImage(self, display):
        # Always the real backing store, which is what makes an
        # under-reporting mode produce an oversized image.
        return self._image(0, 0, self.pixel_w, self.pixel_h)

    def CGDisplayCreateImageForRect(self, display, rect):
        asked = (rect.origin.x, rect.origin.y,
                 rect.size.width, rect.size.height)
        # the real API rounds a fractional rect outward, which is the
        # trap the backend exists to avoid -- so refuse to model it
        assert all(v == int(v) for v in asked), \
            "rect must be whole points, got %r" % (asked,)
        scale_x = self.pixel_w / self.point_w
        scale_y = self.pixel_h / self.point_h
        return self._image(int(asked[0] * scale_x), int(asked[1] * scale_y),
                           int(asked[2] * scale_x), int(asked[3] * scale_y))

    def CGImageGetWidth(self, image):
        return self._meta[image].w

    def CGImageGetHeight(self, image):
        return self._meta[image].h

    def CGImageGetBytesPerRow(self, image):
        return self._meta[image].stride

    def CGImageGetBitsPerPixel(self, image):
        return 32

    def CGImageGetBitsPerComponent(self, image):
        return self.bits_per_component

    def CGImageGetBitmapInfo(self, image):
        return _K_CG_BITMAP_BYTE_ORDER_32_LITTLE

    def CGImageGetDataProvider(self, image):
        return image

    def CGDataProviderCopyData(self, provider):
        self.copy_data_calls += 1
        return provider

    def CGImageRelease(self, image):
        pass

    # ----- the redraw path -----

    def CGImageGetColorSpace(self, image):
        self.colorspace_queries += 1
        return self.image_colorspace

    def CGColorSpaceCreateDeviceRGB(self):
        self.created_colorspaces.append(_DEVICE_RGB)
        return _DEVICE_RGB

    def CGColorSpaceRelease(self, space):
        self.released_colorspaces.append(space)

    def CGBitmapContextCreate(self, data, width, height, bits_per_component,
                              stride, space, info):
        # Pin the destination layout: anything else here and the array
        # handed back would not be the 8-bit BGRA the project promises.
        assert bits_per_component == 8, bits_per_component
        assert info == _BGRA8888, hex(info)
        assert stride == width * 4, (stride, width)
        assert data, "bitmap context created over a NULL buffer"
        if space in self.refuse_colorspaces:
            # What a real CGBitmapContextCreate does with a colour space
            # the pixel format cannot carry: NULL, plus stderr noise.
            return 0
        token = "ctx%d" % (len(self.contexts) + 1)
        self.contexts[token] = (data, width, height, stride)
        self.open_contexts.add(token)
        return token

    def CGContextDrawImage(self, context, rect, image):
        dest, width, height, dest_stride = self.contexts[context]
        meta = self._meta[image]
        # The backend must ask for the image at its own size at the
        # origin; any other rect means it is doing arithmetic that
        # CoreGraphics is supposed to be doing for it.
        assert (rect.origin.x, rect.origin.y) == (0.0, 0.0), rect.origin
        assert (rect.size.width, rect.size.height) == (meta.w, meta.h)
        assert (width, height) == (meta.w, meta.h), (width, height)
        # CoreGraphics reads through its own knowledge of the layout —
        # the parent's stride and this image's offset within it.
        for row in range(height):
            start = meta.data_offset + row * meta.stride
            ctypes.memmove(dest + row * dest_stride,
                           bytes(meta.buf[start:start + width * 4]),
                           width * 4)

    def CGContextRelease(self, context):
        self.open_contexts.discard(context)


class _FakeCoreFoundation:
    """The CoreFoundation symbols, deliberately a separate object.

    Collapsing this into the CoreGraphics fake would hide a symbol
    looked up on the wrong framework handle — which on a real machine
    either raises or resolves with no argtypes set and truncates a
    pointer to 32 bits.
    """

    def __init__(self, cg):
        self._cg = cg

    def CFDataGetBytePtr(self, data):
        return ctypes.addressof(self._cg._buffers[data])

    def CFDataGetLength(self, data):
        return len(self._cg._buffers[data])

    def CFRelease(self, data):
        pass


def _backend(point_w=20, point_h=15, row_pad=0,
             mode_pixel_w=None, mode_pixel_h=None, provider_pad=0,
             parent_margin=0):
    """A MacosBackend wired to fakes, bypassing framework loading."""
    screen = numpy.random.default_rng(0).integers(
        0, 256, (30, 40, 4), dtype=numpy.uint8)
    cg = _FakeCoreGraphics(screen, point_w, point_h, row_pad,
                           mode_pixel_w, mode_pixel_h, provider_pad,
                           parent_margin)
    cf = _FakeCoreFoundation(cg)
    backend = object.__new__(MacosBackend)
    backend._cg = cg
    backend._cf = cf
    backend._display = 1
    # the backend builds these with kwargs; the fake only reads attributes
    backend._CGPoint = backend._CGSize = backend._CGRect = SimpleNamespace
    return backend, cg, cf


def _capture(backend, x, y, w, h):
    img = numpy.zeros((h, w, 4), numpy.uint8)
    backend.screenshot(x, y, img)
    return img


def test_resolution_reports_pixels_not_points():
    """The original bug: resolution() returned points, capture got pixels."""
    backend, _, _ = _backend()
    assert backend.resolution() == (40, 30)


@pytest.mark.parametrize("row_pad", [0, 12])
@pytest.mark.parametrize("x, y, w, h", [
    (0, 0, 40, 30),      # full screen, via CGDisplayCreateImage
    (0, 0, 8, 6),
    (7, 5, 11, 9),       # odd offset and size -> sub-rectangle copy
    (8, 5, 8, 9),        # off_x == 0 but off_y == 1: the off_y-only path
    (39, 29, 1, 1),
])
def test_capture_matches_the_backing_store(x, y, w, h, row_pad):
    backend, cg, _ = _backend(row_pad=row_pad)
    img = _capture(backend, x, y, w, h)
    assert img.shape == (h, w, 4)
    assert numpy.array_equal(img, cg.screen[y:y + h, x:x + w])


def test_capture_matches_without_scaling():
    backend, cg, _ = _backend(point_w=40, point_h=30)
    assert numpy.array_equal(_capture(backend, 7, 5, 11, 9),
                             cg.screen[5:14, 7:18])


def test_undersized_image_is_rejected_rather_than_over_read():
    backend, cg, _ = _backend()
    real = cg.CGImageGetWidth
    cg.CGImageGetWidth = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="too small"):
        _capture(backend, 7, 5, 11, 9)


def test_undersized_image_height_is_rejected_rather_than_over_read():
    """The height half of the guard; only the width half was covered."""
    backend, cg, _ = _backend()
    real = cg.CGImageGetHeight
    cg.CGImageGetHeight = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="too small"):
        _capture(backend, 7, 5, 11, 9)


def test_oversized_full_screen_image_is_rejected():
    """A mode that under-reports must not yield a silent top-left crop.

    Models a virtual/Sidecar-style display whose mode dict omits the
    HiDPI pixel fields, so the mode claims 20x16 while the capture comes
    back as the real 40x30 backing store. A lower-bound size check
    passes that happily and returns the top-left corner.
    """
    backend, _, _ = _backend(mode_pixel_w=20, mode_pixel_h=16)
    assert backend.resolution() == (20, 16)
    with pytest.raises(RuntimeError, match="does not match"):
        _capture(backend, 0, 0, 20, 16)


def test_a_longer_provider_than_expected_is_redrawn_not_refused():
    """Issue #46: refusing an over-allocated provider broke real Macs.

    The extent is the only handle on whether the bytes belong to this
    image alone, so an unexplained one must not be *read* — but it must
    not be fatal either, since CGImageCreate allows it outright.
    """
    backend, cg, cf = _backend()
    real = cf.CFDataGetLength
    cf.CFDataGetLength = lambda data: real(data) + 4096
    assert numpy.array_equal(_capture(backend, 7, 5, 11, 9),
                             cg.screen[5:14, 7:18])
    assert cg.contexts, "should have fallen back to a bitmap context"


def test_stride_narrower_than_a_row_is_rejected():
    """Reported separately from the extent, so the message cannot lie.

    Sharing one error with the extent check let it claim the data length
    matched expectations in the very case where it did.
    """
    backend, cg, _ = _backend()
    real = cg.CGImageGetBytesPerRow
    cg.CGImageGetBytesPerRow = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="unsupported row stride"):
        _capture(backend, 7, 5, 11, 9)


@pytest.mark.parametrize("x, y", [
    (float("nan"), 0),
    (0, float("nan")),
    (7.0, 0),
    (0, 7.5),
    (True, 0),
])
def test_non_integer_origin_is_refused_at_the_backend(x, y):
    """nan defeats the region guard: every comparison against it is False.

    A float origin also produces a float source address that
    ctypes.memmove rejects. Screenshot.capture() normalizes integral
    floats before this point, so only direct callers see this.
    """
    backend, _, _ = _backend()
    img = numpy.zeros((6, 8, 4), numpy.uint8)
    with pytest.raises(ValueError, match="must be an integer"):
        backend.screenshot(x, y, img)


def test_short_cfdata_is_rejected_rather_than_over_read():
    """The provider bytes must cover what the stride arithmetic reads.

    A CGImage that CoreGraphics backs with a parent bitmap reports the
    parent's stride while its data starts at the parent's origin; the
    size checks all pass and every row is copied from the wrong place.
    """
    backend, _, cf = _backend()
    real = cf.CFDataGetLength
    cf.CFDataGetLength = lambda data: real(data) - 1
    with pytest.raises(RuntimeError, match="CFData"):
        _capture(backend, 7, 5, 11, 9)


def test_non_8_bit_components_are_rejected():
    """A 32-bpp 10-10-10-2 surface passes every other check."""
    backend, cg, _ = _backend()
    cg.bits_per_component = 10
    with pytest.raises(RuntimeError, match="bits per component"):
        _capture(backend, 0, 0, 8, 6)


@pytest.mark.parametrize("dimension", [
    "mode_pixel_w", "mode_pixel_h", "point_w", "point_h",
])
def test_degenerate_display_mode_raises_a_clear_error(dimension):
    """A zero dimension must not surface as a bare ZeroDivisionError."""
    backend, cg, _ = _backend()
    setattr(cg, dimension, 0)
    with pytest.raises(RuntimeError, match="display mode"):
        backend.screenshot(1, 1, numpy.zeros((2, 2, 4), numpy.uint8))


@pytest.mark.parametrize("x, y, w, h", [
    (-1, 0, 8, 6),
    (0, -1, 8, 6),
    (36, 0, 8, 6),       # runs past the right edge
    (0, 26, 8, 6),       # runs past the bottom edge
    (0, 0, 0, 6),        # empty region
])
def test_out_of_range_region_is_refused_at_the_backend(x, y, w, h):
    """The backend is reachable directly, not only via check_bbox.

    CoreGraphics clips such a rect rather than refusing it, so without
    this the caller gets a confusing "too small" error or, worse,
    silently shifted pixels.
    """
    backend, _, _ = _backend()
    with pytest.raises(ValueError, match="outside the"):
        backend.screenshot(x, y, numpy.zeros((h, w, 4), numpy.uint8))


def test_refresh_re_reads_the_main_display():
    """The display id is latched in __init__; refresh() must re-resolve."""
    backend, cg, _ = _backend()
    assert backend._display == 1
    cg.main_display = 7
    backend.refresh()
    assert backend._display == 7


def test_refresh_rejects_a_vanished_main_display():
    backend, cg, _ = _backend()
    cg.main_display = 0
    with pytest.raises(RuntimeError, match="no main display"):
        backend.refresh()


def test_out_of_bounds_rect_is_not_silently_zero_padded():
    """Guards the fake itself: numpy slicing used to clip these away."""
    backend, cg, _ = _backend()
    with pytest.raises(AssertionError, match="outside the"):
        cg._image(35, 28, 10, 10)


# --------------------------------------------------------------------
# Issue #46: the data provider is allowed to be bigger than the image.
#
# On the reported machine — 1920x1080, a 7680-byte stride — the CFData
# is 8306688 bytes where the pixels need 8294400. The difference is
# 12288, which is exactly what rounding 8294400 up to a whole 16 KiB
# arm64 page costs. Reading such a provider from offset 0 happens to be
# right; reading a *parent-bitmap* one, which is over-allocated in the
# same undetectable way, is not. So over-allocation gets redrawn.
# --------------------------------------------------------------------

# 8294400 -> 8306688, scaled down to this 40x30 screen: any pad that is
# neither zero nor a whole number of rows.
PAGE_PAD = 12288


def test_the_reported_extent_is_a_16k_page_rounding():
    """Documents the diagnosis the fallback is built around."""
    pixels = 1920 * 1080 * 4
    page = 16 * 1024                      # arm64 macOS page size
    assert pixels == 8294400
    assert -(-pixels // page) * page == 8306688


@pytest.mark.parametrize("row_pad", [0, 12])
@pytest.mark.parametrize("x, y, w, h", [
    (0, 0, 40, 30),      # full screen — the shape issue #46 reports
    (0, 0, 8, 6),
    (7, 5, 11, 9),       # odd offset and size -> sub-rectangle copy
    (39, 29, 1, 1),
])
def test_over_allocated_provider_captures_the_right_pixels(x, y, w, h,
                                                           row_pad):
    """The reported failure, now a capture — with the pixels checked.

    ``row_pad`` covers a stride wider than width*4 on the same path, so
    the redraw's tight destination stride is exercised against a padded
    source.
    """
    backend, cg, _ = _backend(row_pad=row_pad, provider_pad=PAGE_PAD)
    img = _capture(backend, x, y, w, h)
    assert numpy.array_equal(img, cg.screen[y:y + h, x:x + w])
    assert cg.contexts, "an unexplained extent must not be read directly"


@pytest.mark.parametrize("x, y, w, h", [
    (0, 0, 40, 30),
    (7, 5, 11, 9),
])
def test_an_exact_provider_still_takes_the_direct_read(x, y, w, h):
    """The fast path is the point; only anomalies pay for a redraw."""
    backend, cg, _ = _backend()
    assert numpy.array_equal(_capture(backend, x, y, w, h),
                             cg.screen[y:y + h, x:x + w])
    assert cg.contexts == {}, "no redraw should have been needed"
    assert cg.copy_data_calls == 1


def test_sub_image_of_a_parent_bitmap_is_not_read_from_offset_zero():
    """The hazard the extent check was standing in for, met head on.

    The image is a window onto a larger bitmap: the stride is the
    parent's and the byte pointer aims at the *parent's* top-left. The
    redraw asks CoreGraphics where the pixels really are; reading from
    offset 0 would return a neighbouring region instead.
    """
    backend, cg, cf = _backend(parent_margin=2)
    img = _capture(backend, 7, 5, 11, 9)
    assert numpy.array_equal(img, cg.screen[5:14, 7:18])

    # And the fake really does pose the hazard: offset 0 with the
    # reported stride is a different picture, so the test above cannot
    # pass by accident.
    token = max(cg._meta)
    meta = cg._meta[token]
    assert meta.data_offset > 0
    naive = numpy.frombuffer(
        bytes(meta.buf[:meta.stride * meta.h]), dtype=numpy.uint8
    ).reshape(meta.h, meta.stride)[1:10, 4:48].reshape(9, 11, 4)
    assert not numpy.array_equal(naive, cg.screen[5:14, 7:18])


def test_the_redraw_verdict_is_reached_once_not_per_frame():
    """CGDataProviderCopyData copies the whole frame; do not re-pay it."""
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    for _ in range(3):
        assert numpy.array_equal(_capture(backend, 0, 0, 40, 30), cg.screen)
    assert cg.copy_data_calls == 1, "only the probing frame should copy"
    assert len(cg.contexts) == 3


def test_refresh_re_probes_the_provider():
    """A different main display can have a different image layout."""
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    _capture(backend, 0, 0, 40, 30)
    assert backend._draw_via_bitmap is True
    backend.refresh()
    assert backend._draw_via_bitmap is False
    _capture(backend, 0, 0, 40, 30)
    assert cg.copy_data_calls == 2


def test_a_short_provider_is_refused_rather_than_redrawn():
    """Under-length is a different animal from over-length.

    Over-allocation is legal and merely unreadable; a provider shorter
    than bytesPerRow*height contradicts CGImageCreate itself, so the
    image is not what we think it is and no fallback can repair that.
    Reading it would run off the end of the buffer.
    """
    backend, cg, cf = _backend()
    real = cf.CFDataGetLength
    cf.CFDataGetLength = lambda data: real(data) - 1
    with pytest.raises(RuntimeError, match="truncated provider"):
        _capture(backend, 7, 5, 11, 9)
    assert cg.contexts == {}, "a broken image must not be quietly redrawn"
    assert backend._draw_via_bitmap is False


def test_redraw_uses_the_images_own_colour_space():
    """Matching spaces keep the redraw a pure layout conversion.

    Drawing into a device-RGB context colour-matches out of a
    wide-gamut profile, which would hand back different values than the
    direct read does for the same screen.
    """
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    _capture(backend, 0, 0, 40, 30)
    assert cg.created_colorspaces == []
    # Get rule: the image's colour space is not ours to release.
    assert cg.released_colorspaces == []


@pytest.mark.parametrize("image_colorspace, refused", [
    (_IMAGE_RGB, {_IMAGE_RGB}),   # e.g. an EDR profile a bitmap won't take
    (0, set()),                   # CGImageGetColorSpace returned NULL
])
def test_redraw_falls_back_to_device_rgb(image_colorspace, refused):
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    cg.image_colorspace = image_colorspace
    cg.refuse_colorspaces = refused
    assert numpy.array_equal(_capture(backend, 0, 0, 40, 30), cg.screen)
    assert cg.created_colorspaces == [_DEVICE_RGB]
    # Create rule: this one *is* ours, and must not leak.
    assert cg.released_colorspaces == [_DEVICE_RGB]


def test_a_refused_colour_space_is_not_retried_every_frame():
    """A real refusal also writes to stderr; once is plenty."""
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    cg.refuse_colorspaces = {_IMAGE_RGB}
    for _ in range(3):
        _capture(backend, 0, 0, 40, 30)
    assert cg.colorspace_queries == 1
    assert cg.created_colorspaces == [_DEVICE_RGB] * 3


def test_a_bitmap_context_that_cannot_be_made_is_reported():
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    cg.refuse_colorspaces = {_IMAGE_RGB, _DEVICE_RGB}
    with pytest.raises(RuntimeError, match="CGBitmapContextCreate"):
        _capture(backend, 0, 0, 40, 30)
    # even on the failing path the created colour space is released
    assert cg.released_colorspaces == [_DEVICE_RGB]


def test_the_redraw_releases_its_context():
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    for _ in range(3):
        _capture(backend, 0, 0, 40, 30)
    assert cg.open_contexts == set()


def test_the_redraw_buffer_is_reused_across_frames():
    """A per-frame 8 MB allocation on the affected path is avoidable."""
    backend, cg, _ = _backend(provider_pad=PAGE_PAD)
    _capture(backend, 0, 0, 40, 30)
    first = backend._scratch
    _capture(backend, 0, 0, 40, 30)
    assert backend._scratch is first
    # a different size has to reallocate, and must not reuse stale rows
    assert numpy.array_equal(_capture(backend, 7, 5, 11, 9),
                             cg.screen[5:14, 7:18])
