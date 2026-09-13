"""Issue #46's redraw path, against real CoreGraphics on real macOS.

``tests/test_macos_backend.py`` already covers the decision logic with a
fake ctypes layer, including an over-allocated provider (``provider_pad``).
What it cannot cover is the redraw itself: ``CGBitmapContextCreate`` and
``CGContextDrawImage`` are only ever stubs there, because that suite runs
on Linux.

So the machinery the fix actually relies on has never executed. It cannot
be reached from a capture on CI either — measured on the arm64 runner, the
provider comes back exactly the size of its rows for the full display, for
fourteen sub-region sizes including several at fractional page counts,
and even after resizing the display to the 1920x1080 of the report (same
8294400 bytes, same 506.25 pages of 16 KiB, still ``excess=0``). The
padding comes from a real display's backing store, which a headless
virtual display does not have.

What *can* be done here is to build the input by hand: a real ``CGImage``
whose provider is deliberately longer than its rows need, handed to the
backend in place of a captured one. Everything downstream — the extent
check, the redraw, the copy — is then the real thing on the real platform.
"""
import ctypes
import sys

import numpy
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="real CoreGraphics; macOS only"
)

if sys.platform == "darwin":  # keep the import off Linux collection
    from fastgrab.backends.macos import (  # noqa: E402
        _BGRA_BITMAP_INFO, MacosBackend, _load_frameworks,
    )

PAD = 16384  # one arm64 page, the excess issue #46 reported


def _extra_symbols(cg, cf):
    """Declare the CoreGraphics calls only this test needs."""
    cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                               ctypes.c_long]
    cf.CFDataCreate.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None
    cg.CGDataProviderCreateWithCFData.argtypes = [ctypes.c_void_p]
    cg.CGDataProviderCreateWithCFData.restype = ctypes.c_void_p
    cg.CGImageCreate.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t,      # width, height
        ctypes.c_size_t, ctypes.c_size_t,      # bits/component, bits/pixel
        ctypes.c_size_t,                       # bytes per row
        ctypes.c_void_p, ctypes.c_uint32,      # colour space, bitmap info
        ctypes.c_void_p, ctypes.c_void_p,      # provider, decode
        ctypes.c_bool, ctypes.c_int,           # interpolate, intent
    ]
    cg.CGImageCreate.restype = ctypes.c_void_p


def _pattern(width, height):
    """A BGRA image whose every pixel encodes its own coordinates.

    A redraw that flips, shifts or transposes the image produces wrong
    values here rather than merely a differently-shaped array.
    """
    # Reduced mod 256 in a wide dtype before casting: a display is taller
    # than 255 pixels, and numpy 2 raises rather than wrapping.
    rows = numpy.arange(height).reshape(height, 1)
    cols = numpy.arange(width).reshape(1, width)
    out = numpy.zeros((height, width, 4), dtype=numpy.uint8)
    out[:, :, 0] = (cols % 256).astype(numpy.uint8)          # B across
    out[:, :, 1] = (rows % 256).astype(numpy.uint8)          # G down
    out[:, :, 2] = ((cols + rows) % 256).astype(numpy.uint8)  # R both
    out[:, :, 3] = 255
    return out


def _padded_image(cg, cf, width, height):
    """A real CGImage backed by a provider PAD bytes longer than its rows."""
    stride = width * 4
    needed = stride * height
    payload = _pattern(width, height).tobytes()
    assert len(payload) == needed
    # The tail is what the direct read would wrongly treat as pixels.
    blob = payload + (b"\xa5" * PAD)

    data = cf.CFDataCreate(None, blob, len(blob))
    assert data, "CFDataCreate failed"
    provider = cg.CGDataProviderCreateWithCFData(data)
    assert provider, "CGDataProviderCreateWithCFData failed"
    space = cg.CGColorSpaceCreateDeviceRGB()
    assert space, "CGColorSpaceCreateDeviceRGB failed"
    image = cg.CGImageCreate(width, height, 8, 32, stride, space,
                             _BGRA_BITMAP_INFO, provider, None, False, 0)
    assert image, "CGImageCreate failed"
    cg.CGColorSpaceRelease(space)
    return image, needed


class _ImageSwap:
    """The real cg, with the two capture calls returning our image."""

    def __init__(self, cg, image):
        self._cg = cg
        self._image = image
        self.captures = 0

    def __getattr__(self, name):
        return getattr(self._cg, name)

    def CGDisplayCreateImage(self, _display):
        self.captures += 1
        return self._image

    def CGDisplayCreateImageForRect(self, _display, _rect):
        self.captures += 1
        return self._image


@pytest.fixture
def backend_over_padded_provider():
    cg, cf, _CGPoint, _CGSize, _CGRect = _load_frameworks()
    _extra_symbols(cg, cf)
    backend = MacosBackend()
    width, height = backend.resolution()
    image, needed = _padded_image(cg, cf, width, height)
    swap = _ImageSwap(cg, image)
    backend._cg = swap
    return backend, swap, width, height, needed


def test_the_padded_provider_is_seen_as_oversized(backend_over_padded_provider):
    """The premise: the image really does carry a longer provider."""
    backend, swap, width, height, needed = backend_over_padded_provider
    cg, cf = swap._cg, backend._cf
    image = swap._image
    held = cf.CFDataGetLength(cg.CGDataProviderCopyData(
        cg.CGImageGetDataProvider(image)))
    assert held == needed + PAD, (
        "the test's own image is not padded, so it proves nothing"
    )


def test_an_oversized_provider_is_redrawn_not_read(backend_over_padded_provider):
    """The whole point of #47, on the platform it was written for.

    Read directly, the trailing 0xA5 would shift into the picture and the
    last rows would come back as filler.
    """
    backend, swap, width, height, _needed = backend_over_padded_provider
    out = numpy.zeros((height, width, 4), numpy.uint8)
    backend.screenshot(0, 0, out)

    assert swap.captures >= 1
    assert backend._draw_via_bitmap is True, (
        "the backend read the provider directly instead of redrawing"
    )
    expected = _pattern(width, height)
    # Alpha is the one channel the redraw is allowed to change: the
    # context is premultiplied-first, and the source is fully opaque.
    assert numpy.array_equal(out[:, :, :3], expected[:, :, :3]), (
        "redrawn pixels do not match the pattern that went in"
    )
    assert not (out == 0xA5).all(axis=2).any(), "provider padding reached the image"


def test_the_redraw_latch_persists(backend_over_padded_provider):
    """Once seen, later captures skip the provider probe entirely."""
    backend, _swap, width, height, _needed = backend_over_padded_provider
    out = numpy.zeros((height, width, 4), numpy.uint8)
    backend.screenshot(0, 0, out)
    assert backend._draw_via_bitmap is True
    again = numpy.zeros((height, width, 4), numpy.uint8)
    backend.screenshot(0, 0, again)
    assert numpy.array_equal(out, again)
