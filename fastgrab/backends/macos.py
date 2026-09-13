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

Getting the pixels *out* of the returned ``CGImage`` has two paths.
Normally the image's data provider holds exactly its own rows, and they
are read straight out of it. When it does not — the provider is allowed
to be larger than ``bytesPerRow * height``, and on Apple Silicon a
capture arrives padded out to the 16 KiB page — the image is redrawn
into a bitmap context whose layout we chose, which is the only way to
place the pixels without guessing. See :meth:`MacosBackend.screenshot`.

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
import numbers

import numpy

from .base import BaseBackend


# CGImageAlphaInfo bits — keep in sync with CoreGraphics/CGImage.h
_K_CG_BITMAP_BYTE_ORDER_MASK = 0x7000
_K_CG_BITMAP_BYTE_ORDER_32_LITTLE = 2 << 12  # kCGBitmapByteOrder32Little
_K_CG_IMAGE_ALPHA_PREMULTIPLIED_FIRST = 2  # kCGImageAlphaPremultipliedFirst

# The layout we ask a bitmap context for: alpha "first" within the 32-bit
# word, and 32-little reverses that word's bytes, so memory reads B, G,
# R, A -- fastgrab's cross-platform contract. Screen captures are opaque,
# so premultiplication is a no-op and only fixes the alpha byte at 255.
_BGRA_BITMAP_INFO = (_K_CG_IMAGE_ALPHA_PREMULTIPLIED_FIRST
                     | _K_CG_BITMAP_BYTE_ORDER_32_LITTLE)


def _ceil_div(numerator, denominator):
    """Ceiling of ``numerator / denominator`` for non-negative integers."""
    return -(-numerator // denominator)


def _snap_axis(start, length, pixels, points):
    """Snap one axis of a device-pixel span out to whole points.

    ``pixels``/``points`` are that axis' extent in each unit. Returns
    ``(pt_start, pt_length, offset)``: where to ask in points, how much
    to ask for in points, and how far into the returned image the
    requested span begins, in pixels.
    """
    pt_start = (start * points) // pixels
    pt_end = _ceil_div((start + length) * points, pixels)
    return pt_start, pt_end - pt_start, start - (pt_start * pixels) // points


def _snap_rect_to_points(x, y, w, h, pixel_w, pixel_h, point_w, point_h):
    """Map a device-pixel rect to the whole-point rect containing it.

    ``CGDisplayCreateImageForRect`` takes points but returns pixels, and
    rounds a fractional rect *outward* — 400.5 points yields 802 pixels,
    not 801 — so round outward ourselves.

    Returns ``(pt_x, pt_y, pt_w, pt_h, off_x, off_y)``: the rect to ask
    for, in points, and the pixel offset of the requested region inside
    the returned image.
    """
    pt_x, pt_w, off_x = _snap_axis(x, w, pixel_w, point_w)
    pt_y, pt_h, off_y = _snap_axis(y, h, pixel_h, point_h)
    return (pt_x, pt_y, pt_w, pt_h, off_x, off_y)


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

    # The redraw path (see MacosBackend._bitmap_copy).
    cg.CGImageGetColorSpace.argtypes = [ctypes.c_void_p]
    cg.CGImageGetColorSpace.restype = ctypes.c_void_p
    cg.CGColorSpaceCreateDeviceRGB.argtypes = []
    cg.CGColorSpaceCreateDeviceRGB.restype = ctypes.c_void_p
    cg.CGColorSpaceRelease.argtypes = [ctypes.c_void_p]
    cg.CGColorSpaceRelease.restype = None
    cg.CGBitmapContextCreate.argtypes = [
        ctypes.c_void_p,   # data
        ctypes.c_size_t,   # width
        ctypes.c_size_t,   # height
        ctypes.c_size_t,   # bitsPerComponent
        ctypes.c_size_t,   # bytesPerRow
        ctypes.c_void_p,   # space
        ctypes.c_uint32,   # bitmapInfo
    ]
    cg.CGBitmapContextCreate.restype = ctypes.c_void_p
    cg.CGContextDrawImage.argtypes = [ctypes.c_void_p, CGRect,
                                      ctypes.c_void_p]
    cg.CGContextDrawImage.restype = None
    cg.CGContextRelease.argtypes = [ctypes.c_void_p]
    cg.CGContextRelease.restype = None

    # ----- CoreFoundation types -----
    cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    cf.CFDataGetBytePtr.restype = ctypes.c_void_p
    cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
    cf.CFDataGetLength.restype = ctypes.c_long
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None

    return cg, cf, CGPoint, CGSize, CGRect


class MacosBackend(BaseBackend):
    # Class-level so a backend built without __init__ (the tests do this
    # to bypass framework loading) still has them.
    #
    # _draw_via_bitmap latches once a provider is seen that cannot be
    # read directly, so later frames skip the CGDataProviderCopyData
    # probe entirely. It only ever turns on: the redraw is correct for
    # every layout, so a needless one costs speed, never pixels.
    _draw_via_bitmap = False
    # Whether to keep offering the image's own colour space to
    # CGBitmapContextCreate; cleared for good the first time one is
    # refused, so a display with such a profile does not pay for a
    # failed create -- and its CoreGraphics stderr complaint -- per
    # frame.
    _bitmap_space_from_image = True
    # Destination for the redraw, reused while the size holds.
    _scratch = None

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
        # A different display can have a different image layout, so let
        # the next capture re-probe rather than inherit this one's verdict.
        self._draw_via_bitmap = False
        self._bitmap_space_from_image = True

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

        # Reject a non-integer origin before the comparisons below, not
        # after: every one of them is False for nan, so nan would sail
        # through the region guard and reach the snapping arithmetic,
        # and a float origin produces a float source address that
        # ctypes.memmove rejects. Screenshot.capture normalizes integral
        # floats to int before this point, so nothing valid is refused.
        for _name, _value in (("x", x), ("y", y)):
            if isinstance(_value, bool) or not isinstance(
                    _value, numbers.Integral):
                raise ValueError(
                    "{} must be an integer number of device pixels, got "
                    "{!r}".format(_name, _value)
                )
        x, y = int(x), int(y)

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

            if row_stride < img_w * 4:
                # Not a layout any fallback rescues: a 32-bit row cannot
                # be shorter than four bytes a pixel, so one of these
                # numbers does not mean what we think it means.
                raise RuntimeError(
                    "unsupported row stride: CGImage reports {} bytes "
                    "per row for a {}-pixel-wide 32-bit image, which "
                    "needs at least {}. Please open an issue."
                    .format(row_stride, img_w, img_w * 4)
                )

            # Two ways out of a CGImage, and which one is safe depends
            # on the provider.
            #
            # Reading the provider bytes directly assumes they start at
            # this image's own top-left with the stride reported above.
            # A CGImage backed by a *parent* bitmap breaks that: the
            # stride is the parent's, the data begins at the parent's
            # origin, and offset 0 is somebody else's pixels.
            #
            # Only an exact extent — the provider holding precisely this
            # image's rows — makes the direct read sound, and it is not
            # something CoreGraphics promises: CGImageCreate takes the
            # provider and the extent as separate inputs and requires
            # only that the buffer hold at least bytesPerRow*height, so
            # a larger provider is legal. It is also *common*. Issue #46
            # reports a 1920x1080 capture whose CFData is 8306688 bytes
            # rather than 8294400 — the pixels rounded up to a whole
            # 16 KiB page, the arm64 page size. A larger provider cannot
            # be told apart from a shared parent by size alone, so
            # anything but an exact match goes the layout-agnostic way
            # rather than guessing (or, as before, refusing to run).
            #
            # A *short* provider stays fatal. That one contradicts
            # CGImageCreate's own minimum, so our reading of the image
            # is wrong in some way no fallback repairs, and the direct
            # read would run off the end of the buffer.
            copied = False
            if not self._draw_via_bitmap:
                provider = self._cg.CGImageGetDataProvider(image)
                cf_data = self._cg.CGDataProviderCopyData(provider)
                if not cf_data:
                    raise RuntimeError("CGDataProviderCopyData returned NULL")

                try:
                    src_ptr = self._cf.CFDataGetBytePtr(cf_data)
                    if not src_ptr:
                        raise RuntimeError("CFDataGetBytePtr returned NULL")

                    data_len = self._cf.CFDataGetLength(cf_data)
                    minimum = row_stride * img_h
                    if data_len < minimum:
                        raise RuntimeError(
                            "truncated provider: CFData holds {} bytes for a "
                            "{}x{} image with a {}-byte row stride, which "
                            "needs at least {}. Reading it would run past "
                            "the end of the buffer. Please open an issue."
                            .format(data_len, img_w, img_h, row_stride,
                                    minimum)
                        )

                    if data_len == minimum:
                        # One strided view over the provider bytes,
                        # sliced to the requested region and copied in a
                        # single C-level pass. Row padding (Apple often
                        # aligns to 16 bytes) and a snapped left edge are
                        # both just slicing here; done with per-row
                        # memmoves they cost an interpreter round trip
                        # per row, and a snapped left edge is the common
                        # case for sub-rect captures on a 2x display.
                        src = numpy.frombuffer(
                            (ctypes.c_ubyte * minimum).from_address(src_ptr),
                            dtype=numpy.uint8,
                        ).reshape(img_h, row_stride)
                        img[:] = src[
                            off_y:off_y + h, off_x * 4:(off_x + w) * 4
                        ].reshape(h, w, 4)
                        copied = True
                    else:
                        # Latch, so the next frame goes straight to the
                        # redraw instead of paying for this copy to
                        # re-learn the same answer.
                        self._draw_via_bitmap = True
                finally:
                    self._cf.CFRelease(cf_data)

            if not copied:
                src = self._bitmap_copy(image, img_w, img_h)
                img[:] = src[
                    off_y:off_y + h, off_x * 4:(off_x + w) * 4
                ].reshape(h, w, 4)
        finally:
            self._cg.CGImageRelease(image)

    def _bitmap_copy(self, image, img_w, img_h):
        """Redraw ``image`` into a bitmap whose layout we chose.

        ``CGContextDrawImage`` is the layout-agnostic way to get pixels
        out of a ``CGImage``: CoreGraphics resolves the provider, the
        stride and any parent-bitmap origin itself, so none of them can
        be guessed wrong here, and the destination is BGRA8 because we
        asked for BGRA8.

        Returns an ``(img_h, img_w * 4)`` uint8 array — the same shape
        the direct read produces, so the caller slices it identically.
        The buffer is reused across captures of the same size.
        """
        stride = img_w * 4
        scratch = self._scratch
        if scratch is None or scratch.shape != (img_h, stride):
            # zeros rather than empty: were a draw ever to fail without
            # saying so, the caller gets black, not heap contents.
            scratch = numpy.zeros((img_h, stride), numpy.uint8)
            self._scratch = scratch

        # Offer the image's own colour space first. Source and
        # destination spaces matching is what keeps this a pure layout
        # conversion — CoreGraphics colour-matches between differing
        # spaces, which would quietly hand back different pixel values
        # than the direct read does on a wide-gamut display. Device RGB
        # is the fallback for a space a bitmap context will not accept
        # (an extended-range/EDR profile needs float components), and it
        # does convert.
        space = None
        own_space = False
        if self._bitmap_space_from_image:
            # Get rule: that reference is the image's, not ours to release.
            space = self._cg.CGImageGetColorSpace(image)
        context = None
        if space:
            context = self._cg.CGBitmapContextCreate(
                scratch.ctypes.data, img_w, img_h, 8, stride, space,
                _BGRA_BITMAP_INFO)
            if not context:
                self._bitmap_space_from_image = False
        if not context:
            # Create rule: we own this one and must release it.
            space = self._cg.CGColorSpaceCreateDeviceRGB()
            own_space = True
            if space:
                context = self._cg.CGBitmapContextCreate(
                    scratch.ctypes.data, img_w, img_h, 8, stride, space,
                    _BGRA_BITMAP_INFO)

        try:
            if not context:
                raise RuntimeError(
                    "CGBitmapContextCreate returned NULL for a {}x{} BGRA8 "
                    "bitmap; fastgrab cannot convert this CGImage. Please "
                    "open an issue.".format(img_w, img_h)
                )
            # Exactly the image's own size, so the transform is the
            # identity and nothing is resampled. A bitmap context's
            # first row in memory is its top row, so this lands upright
            # despite the y-up coordinate space.
            rect = self._CGRect(
                origin=self._CGPoint(x=0.0, y=0.0),
                size=self._CGSize(width=float(img_w), height=float(img_h)),
            )
            self._cg.CGContextDrawImage(context, rect, image)
        finally:
            if context:
                self._cg.CGContextRelease(context)
            if own_space and space:
                self._cg.CGColorSpaceRelease(space)

        return scratch
