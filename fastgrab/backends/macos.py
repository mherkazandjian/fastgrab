"""macOS screen capture backend (CoreGraphics ``CGDisplayCreateImage``).

Pure-Python: no extras, no PyObjC. We talk to the CoreGraphics
framework directly through :mod:`ctypes`, so a default
``pip install fastgrab`` works on a fresh macOS install with nothing
but Python + numpy.

V1 captures from the **main** display only (``CGMainDisplayID``).
Multi-display capture via ``CGGetActiveDisplayList`` is a future
``Screenshot(display=N)`` extension.

Byte order matches the rest of fastgrab: ``CGDisplayCreateImage``
returns 32-bit-per-pixel little-endian BGRA on both Intel and Apple
Silicon, so the captured numpy array is BGRA — same contract as X11
and wlr. The bitmap-info bits are double-checked at runtime; an
unexpected layout raises :class:`RuntimeError` with a clear message
rather than silently scrambling channels.

**TCC permission caveat:** macOS 10.15+ requires the running app to
have *Screen Recording* permission via System Settings → Privacy
& Security. CI runners (``macos-latest``) typically have this granted
for the ``runner`` user. On a developer machine, the first capture
attempt may return a uniformly-black frame; that is a TCC denial,
not a fastgrab bug.
"""
from __future__ import annotations

import ctypes
import ctypes.util

from .base import BaseBackend


# CGImageAlphaInfo bits — keep in sync with CoreGraphics/CGImage.h
_K_CG_BITMAP_BYTE_ORDER_MASK = 0x7000
_K_CG_BITMAP_BYTE_ORDER_32_LITTLE = 2 << 12  # kCGBitmapByteOrder32Little


def _load_frameworks():
    cg_path = ctypes.util.find_library("CoreGraphics")
    cf_path = ctypes.util.find_library("CoreFoundation")
    if not cg_path or not cf_path:
        # find_library should always succeed on macOS for these — bail loudly.
        raise RuntimeError(
            "could not locate CoreGraphics / CoreFoundation frameworks "
            "(find_library returned: cg={!r} cf={!r})".format(cg_path, cf_path)
        )
    cg = ctypes.CDLL(cg_path)
    cf = ctypes.CDLL(cf_path)

    # ----- CoreGraphics types -----
    # CGDirectDisplayID is uint32; CGFloat is double on 64-bit (the only
    # macOS we support); pointers are c_void_p.
    cg.CGMainDisplayID.argtypes = []
    cg.CGMainDisplayID.restype = ctypes.c_uint32

    cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
    cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
    cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
    cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t

    class CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    class CGSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

    class CGRect(ctypes.Structure):
        _fields_ = [("origin", CGPoint), ("size", CGSize)]

    cg.CGDisplayCreateImage.argtypes = [ctypes.c_uint32]
    cg.CGDisplayCreateImage.restype = ctypes.c_void_p
    cg.CGDisplayCreateImageForRect.argtypes = [ctypes.c_uint32, CGRect]
    cg.CGDisplayCreateImageForRect.restype = ctypes.c_void_p

    cg.CGImageGetWidth.argtypes = [ctypes.c_void_p]
    cg.CGImageGetWidth.restype = ctypes.c_size_t
    cg.CGImageGetHeight.argtypes = [ctypes.c_void_p]
    cg.CGImageGetHeight.restype = ctypes.c_size_t
    cg.CGImageGetBytesPerRow.argtypes = [ctypes.c_void_p]
    cg.CGImageGetBytesPerRow.restype = ctypes.c_size_t
    cg.CGImageGetBitsPerPixel.argtypes = [ctypes.c_void_p]
    cg.CGImageGetBitsPerPixel.restype = ctypes.c_size_t
    cg.CGImageGetBitmapInfo.argtypes = [ctypes.c_void_p]
    cg.CGImageGetBitmapInfo.restype = ctypes.c_uint32
    cg.CGImageGetDataProvider.argtypes = [ctypes.c_void_p]
    cg.CGImageGetDataProvider.restype = ctypes.c_void_p
    cg.CGImageRelease.argtypes = [ctypes.c_void_p]
    cg.CGImageRelease.restype = None

    cg.CGDataProviderCopyData.argtypes = [ctypes.c_void_p]
    cg.CGDataProviderCopyData.restype = ctypes.c_void_p

    # ----- CoreFoundation types -----
    cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    cf.CFDataGetBytePtr.restype = ctypes.c_void_p
    cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
    cf.CFDataGetLength.restype = ctypes.c_long
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None

    return cg, cf, CGPoint, CGSize, CGRect


class MacosBackend(BaseBackend):
    def __init__(self):
        cg, cf, CGPoint, CGSize, CGRect = _load_frameworks()
        self._cg = cg
        self._cf = cf
        self._CGPoint = CGPoint
        self._CGSize = CGSize
        self._CGRect = CGRect
        self._display = cg.CGMainDisplayID()
        if self._display == 0:
            raise RuntimeError("CGMainDisplayID returned 0; no main display?")

    # -------- BaseBackend API --------

    def resolution(self):
        w = self._cg.CGDisplayPixelsWide(self._display)
        h = self._cg.CGDisplayPixelsHigh(self._display)
        return (int(w), int(h))

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        h, w, _ = img.shape
        full_w, full_h = self.resolution()

        if x == 0 and y == 0 and w == full_w and h == full_h:
            image = self._cg.CGDisplayCreateImage(self._display)
        else:
            rect = self._CGRect(
                origin=self._CGPoint(x=float(x), y=float(y)),
                size=self._CGSize(width=float(w), height=float(h)),
            )
            image = self._cg.CGDisplayCreateImageForRect(self._display, rect)
        if not image:
            raise RuntimeError(
                "CGDisplayCreateImage returned NULL — likely a Screen "
                "Recording (TCC) permission denial. Grant access via "
                "System Settings → Privacy & Security → Screen Recording."
            )

        try:
            bpp = self._cg.CGImageGetBitsPerPixel(image)
            if bpp != 32:
                raise RuntimeError(
                    "expected 32-bpp image from CGDisplayCreateImage, got "
                    "{} bpp".format(bpp)
                )
            info = self._cg.CGImageGetBitmapInfo(image)
            if (info & _K_CG_BITMAP_BYTE_ORDER_MASK) != _K_CG_BITMAP_BYTE_ORDER_32_LITTLE:
                # We've only ever observed 32-little (BGRA) on shipping
                # macOS. If a future system surprises us, fail loudly so a
                # bug report can teach us what to do.
                raise RuntimeError(
                    "unexpected CGImage byte order; bitmap_info=0x{:08x}. "
                    "fastgrab's MacosBackend assumes 32-bit little-endian "
                    "BGRA — please open an issue.".format(info)
                )

            row_stride = self._cg.CGImageGetBytesPerRow(image)
            img_w = self._cg.CGImageGetWidth(image)
            img_h = self._cg.CGImageGetHeight(image)
            if img_w != w or img_h != h:
                raise RuntimeError(
                    "CGImage size {}x{} does not match requested {}x{}; "
                    "Retina scale-factor mismatch?".format(img_w, img_h, w, h)
                )

            provider = self._cg.CGImageGetDataProvider(image)
            cf_data = self._cg.CGDataProviderCopyData(provider)
            if not cf_data:
                raise RuntimeError("CGDataProviderCopyData returned NULL")

            try:
                src_ptr = self._cf.CFDataGetBytePtr(cf_data)
                if not src_ptr:
                    raise RuntimeError("CFDataGetBytePtr returned NULL")
                dst_ptr = img.ctypes.data
                if row_stride == w * 4:
                    # Tightly packed — single memmove.
                    ctypes.memmove(dst_ptr, src_ptr, w * h * 4)
                else:
                    # Stride padding (Apple often aligns to 16 bytes) —
                    # copy row by row.
                    for row in range(h):
                        ctypes.memmove(
                            dst_ptr + row * w * 4,
                            src_ptr + row * row_stride,
                            w * 4,
                        )
            finally:
                self._cf.CFRelease(cf_data)
        finally:
            self._cg.CGImageRelease(image)
