"""Tests for the macOS backend: the points-vs-pixels split on Retina.

The rect arithmetic is a pure function, tested directly. The copy path
memmoves out of a CFData pointer, so it gets a fake ctypes layer — the
only place ``off_x``/``off_y`` ever run, since CI Macs are 1x.
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


@pytest.mark.parametrize("rect", [(0, 0, 8, 6), (7, 5, 11, 9), (39, 29, 1, 1)])
def test_rect_is_unchanged_without_scaling(rect):
    x, y, w, h = rect
    assert _snap_rect_to_points(*rect, **FLAT) == (x, y, w, h, 0, 0)


class _FakeCoreGraphics:
    """The slice of CoreGraphics the backend calls, over a numpy screen."""

    def __init__(self, screen, point_w, point_h, row_pad=0):
        self.screen = screen
        self.pixel_h, self.pixel_w = screen.shape[:2]
        self.point_w, self.point_h = point_w, point_h
        self.row_pad = row_pad
        self._meta = {}
        self._buffers = {}

    def CGDisplayCopyDisplayMode(self, display):
        return 1

    def CGDisplayModeRelease(self, mode):
        pass

    def CGDisplayModeGetWidth(self, mode):
        return self.point_w

    def CGDisplayModeGetHeight(self, mode):
        return self.point_h

    def CGDisplayModeGetPixelWidth(self, mode):
        return self.pixel_w

    def CGDisplayModeGetPixelHeight(self, mode):
        return self.pixel_h

    def _image(self, x, y, w, h):
        stride = w * 4 + self.row_pad
        buf = (ctypes.c_ubyte * (stride * h))()
        for index, row in enumerate(self.screen[y:y + h, x:x + w]):
            ctypes.memmove(ctypes.byref(buf, index * stride), row.tobytes(),
                           w * 4)
        token = len(self._meta) + 1
        self._meta[token] = (w, h, stride)
        self._buffers[token] = buf
        return token

    def CGDisplayCreateImage(self, display):
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
        return self._meta[image][0]

    def CGImageGetHeight(self, image):
        return self._meta[image][1]

    def CGImageGetBytesPerRow(self, image):
        return self._meta[image][2]

    def CGImageGetBitsPerPixel(self, image):
        return 32

    def CGImageGetBitmapInfo(self, image):
        return _K_CG_BITMAP_BYTE_ORDER_32_LITTLE

    def CGImageGetDataProvider(self, image):
        return image

    def CGDataProviderCopyData(self, provider):
        return provider

    def CGImageRelease(self, image):
        pass

    def CFDataGetBytePtr(self, data):
        return ctypes.addressof(self._buffers[data])

    def CFRelease(self, data):
        pass


def _backend(point_w=20, point_h=15, row_pad=0):
    """A MacosBackend wired to fakes, bypassing framework loading."""
    screen = numpy.random.default_rng(0).integers(
        0, 256, (30, 40, 4), dtype=numpy.uint8)
    cg = _FakeCoreGraphics(screen, point_w, point_h, row_pad)
    backend = object.__new__(MacosBackend)
    backend._cg = backend._cf = cg
    backend._display = 1
    # the backend builds these with kwargs; the fake only reads attributes
    backend._CGPoint = backend._CGSize = backend._CGRect = SimpleNamespace
    return backend, cg


def _capture(backend, x, y, w, h):
    img = numpy.zeros((h, w, 4), numpy.uint8)
    backend.screenshot(x, y, img)
    return img


def test_resolution_reports_pixels_not_points():
    """The original bug: resolution() returned points, capture got pixels."""
    backend, _ = _backend()
    assert backend.resolution() == (40, 30)


@pytest.mark.parametrize("row_pad", [0, 12])
@pytest.mark.parametrize("x, y, w, h", [
    (0, 0, 40, 30),      # full screen, via CGDisplayCreateImage
    (0, 0, 8, 6),
    (7, 5, 11, 9),       # odd offset and size -> sub-rectangle copy
    (39, 29, 1, 1),
])
def test_capture_matches_the_backing_store(x, y, w, h, row_pad):
    backend, cg = _backend(row_pad=row_pad)
    img = _capture(backend, x, y, w, h)
    assert img.shape == (h, w, 4)
    assert numpy.array_equal(img, cg.screen[y:y + h, x:x + w])


def test_capture_matches_without_scaling():
    backend, cg = _backend(point_w=40, point_h=30)
    assert numpy.array_equal(_capture(backend, 7, 5, 11, 9),
                             cg.screen[5:14, 7:18])


def test_undersized_image_is_rejected_rather_than_over_read():
    backend, cg = _backend()
    real = cg.CGImageGetWidth
    cg.CGImageGetWidth = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="too small"):
        _capture(backend, 7, 5, 11, 9)
