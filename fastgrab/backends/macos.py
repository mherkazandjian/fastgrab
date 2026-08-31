"""macOS screen capture backend (CoreGraphics ``CGDisplayCreateImage``).

Pure-Python: no extras, no PyObjC. We talk to the CoreGraphics
framework directly through :mod:`ctypes`, so a default
``pip install fastgrab`` works on a fresh macOS install with nothing
but Python + numpy.

V1 captures from the **main** display only (``CGMainDisplayID``).
Multi-display capture via ``CGGetActiveDisplayList`` is a future
``Screenshot(display=N)`` extension.

**Coordinates are device pixels**, matching X11 and wlr, and so are
``bbox`` rectangles: a 2x panel with an 1800x1169-point desktop reports
and captures 3600x2338.

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


def _snap_rect_to_points(x, y, w, h, pixel_w, pixel_h, point_w, point_h):
    """Map a device-pixel rect to the whole-point rect containing it.

    ``CGDisplayCreateImageForRect`` takes points but returns pixels, and
    rounds a fractional rect *outward* — 400.5 points yields 802 pixels,
    not 801 — so round outward ourselves.

    Returns ``(pt_x, pt_y, pt_w, pt_h, off_x, off_y)``: the rect to ask
    for, in points, and the pixel offset of the requested region inside
    the returned image.
    """
    pt_x0 = (x * point_w) // pixel_w
    pt_y0 = (y * point_h) // pixel_h
    pt_x1 = -((-(x + w) * point_w) // pixel_w)
    pt_y1 = -((-(y + h) * point_h) // pixel_h)
    return (
        pt_x0, pt_y0, pt_x1 - pt_x0, pt_y1 - pt_y0,
        x - (pt_x0 * pixel_w) // point_w,
        y - (pt_y0 * pixel_h) // point_h,
    )


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

    cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
    cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p

    cg.CGDisplayModeGetWidth.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeGetWidth.restype = ctypes.c_size_t
    cg.CGDisplayModeGetHeight.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeGetHeight.restype = ctypes.c_size_t
    cg.CGDisplayModeGetPixelWidth.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeGetPixelWidth.restype = ctypes.c_size_t
    cg.CGDisplayModeGetPixelHeight.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeGetPixelHeight.restype = ctypes.c_size_t
    cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
    cg.CGDisplayModeRelease.restype = None

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
    cg.CGImageGetBitsPerComponent.argtypes = [ctypes.c_void_p]
    cg.CGImageGetBitsPerComponent.restype = ctypes.c_size_t
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

    def refresh(self):
        """Re-read which display is the main one.

        ``CGMainDisplayID`` is resolved once at construction, so
        unplugging a monitor or promoting a different one to main is
        otherwise never noticed.
        """
        display = self._cg.CGMainDisplayID()
        if display == 0:
            raise RuntimeError("CGMainDisplayID returned 0; no main display?")
        self._display = display

    def _display_geometry(self):
        """Return ``(pixel_w, pixel_h, point_w, point_h)``.

        Read fresh on every call, so this backend itself never holds a
        stale mode. Note that the size the *public* API reports is
        cached by :attr:`fastgrab.screenshot.Screenshot.screensize`; a
        mode switch is picked up on the next capture only after calling
        :meth:`fastgrab.screenshot.Screenshot.refresh`.
        """
        mode = self._cg.CGDisplayCopyDisplayMode(self._display)
        if not mode:
            raise RuntimeError("CGDisplayCopyDisplayMode returned NULL")
        try:
            geometry = (
                int(self._cg.CGDisplayModeGetPixelWidth(mode)),
                int(self._cg.CGDisplayModeGetPixelHeight(mode)),
                int(self._cg.CGDisplayModeGetWidth(mode)),
                int(self._cg.CGDisplayModeGetHeight(mode)),
            )
        finally:
            # Core Foundation *copy* rule: we own the +1 reference.
            self._cg.CGDisplayModeRelease(mode)

        if not all(geometry):
            # A mirrored, virtual or sleeping display can report zero for
            # a dimension. Left alone it divides by zero inside the
            # snapping arithmetic, which says nothing about the display.
            raise RuntimeError(
                "display mode reports a zero dimension: pixels={}x{} "
                "points={}x{} — the display may be asleep, mirrored or "
                "virtual.".format(*geometry)
            )

        return geometry

    def resolution(self):
        pixel_w, pixel_h, _, _ = self._display_geometry()
        return (pixel_w, pixel_h)

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        h, w, _ = img.shape
        pixel_w, pixel_h, point_w, point_h = self._display_geometry()

        # Screenshot.check_bbox validates too, but this entry point is
        # reachable directly. CoreGraphics clips a rect that reaches
        # outside the display instead of refusing it, which would land
        # as a confusing "CGImage too small" -- or, if the clipped image
        # is still large enough, as silently shifted pixels.
        if (w <= 0 or h <= 0 or x < 0 or y < 0
                or x > pixel_w - w or y > pixel_h - h):
            raise ValueError(
                "region {}x{} at {},{} is outside the {}x{} display"
                .format(w, h, x, y, pixel_w, pixel_h)
            )

        # A full-screen grab predicts the returned size exactly; a snapped
        # sub-rect legitimately comes back larger than the region asked
        # for, so the two cases get different size checks below.
        whole_screen = x == 0 and y == 0 and w == pixel_w and h == pixel_h
        if whole_screen:
            image = self._cg.CGDisplayCreateImage(self._display)
            off_x = off_y = 0
        else:
            pt_x, pt_y, pt_w, pt_h, off_x, off_y = _snap_rect_to_points(
                x, y, w, h, pixel_w, pixel_h, point_w, point_h
            )
            rect = self._CGRect(
                origin=self._CGPoint(x=float(pt_x), y=float(pt_y)),
                size=self._CGSize(width=float(pt_w), height=float(pt_h)),
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
            bits_per_component = self._cg.CGImageGetBitsPerComponent(image)
            if bits_per_component != 8:
                # 32 bpp is not enough on its own: a wide-gamut/EDR
                # surface is 32-bit 10-10-10-2, which passes the bpp and
                # byte-order checks and copies the right number of bytes
                # while scrambling every channel value.
                raise RuntimeError(
                    "expected 8 bits per component from "
                    "CGDisplayCreateImage, got {} — this looks like a "
                    "wide-gamut/EDR surface, which fastgrab cannot yet "
                    "convert to BGRA8. Please open an issue."
                    .format(bits_per_component)
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
            if whole_screen:
                # The display mode said exactly how big this would be, so
                # anything else means the mode changed under us or this
                # display reports a scale factor we do not understand.
                # Accepting a larger image here would silently return its
                # top-left corner.
                if img_w != pixel_w or img_h != pixel_h:
                    raise RuntimeError(
                        "CGImage {}x{} does not match the display mode's "
                        "{}x{} pixel size — the mode changed mid-capture, "
                        "or this display reports a scale factor fastgrab "
                        "does not understand. Please open an issue."
                        .format(img_w, img_h, pixel_w, pixel_h)
                    )
            elif img_w < off_x + w or img_h < off_y + h:
                # Snapping to whole points legitimately over-provisions,
                # so only a *short* image is wrong here.
                raise RuntimeError(
                    "CGImage {}x{} too small for a {}x{} region at "
                    "offset {},{}".format(img_w, img_h, w, h, off_x, off_y)
                )

            provider = self._cg.CGImageGetDataProvider(image)
            cf_data = self._cg.CGDataProviderCopyData(provider)
            if not cf_data:
                raise RuntimeError("CGDataProviderCopyData returned NULL")

            try:
                src_ptr = self._cf.CFDataGetBytePtr(cf_data)
                if not src_ptr:
                    raise RuntimeError("CFDataGetBytePtr returned NULL")

                # The copy trusts that the provider bytes start at this
                # image's own top-left with the stride reported above. A
                # CGImage backed by a *parent* bitmap breaks that: the
                # stride is the parent's and the data begins at the
                # parent's origin. Bounding the read against the actual
                # length turns that into a loud error instead of rows
                # copied from the wrong place.
                data_len = self._cf.CFDataGetLength(cf_data)
                needed = (off_y + h - 1) * row_stride + (off_x + w) * 4
                if data_len < needed:
                    raise RuntimeError(
                        "CFData holds {} bytes, but a {}x{} region at "
                        "offset {},{} with a {}-byte row stride needs {} "
                        "— the CGImage may be a view into a larger parent "
                        "bitmap. Please open an issue.".format(
                            data_len, w, h, off_x, off_y, row_stride, needed)
                    )

                dst_ptr = img.ctypes.data
                base = src_ptr + off_y * row_stride + off_x * 4
                if off_x == 0 and row_stride == w * 4:
                    # Tightly packed — single memmove.
                    ctypes.memmove(dst_ptr, base, w * h * 4)
                else:
                    # Stride padding (Apple often aligns to 16 bytes) or
                    # a snapped left edge — copy row by row.
                    for row in range(h):
                        ctypes.memmove(
                            dst_ptr + row * w * 4,
                            base + row * row_stride,
                            w * 4,
                        )
            finally:
                self._cf.CFRelease(cf_data)
        finally:
            self._cg.CGImageRelease(image)
