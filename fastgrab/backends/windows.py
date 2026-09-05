"""Windows screen capture backend (BitBlt + DIBSection via ctypes).

Pure-Python: no extras, no ``pywin32``. We talk to ``user32.dll`` and
``gdi32.dll`` directly through :mod:`ctypes` so a default
``pip install fastgrab`` works on a fresh Windows 10 / 11 install with
nothing but Python + numpy.

V1 captures from the **primary** monitor only. Multi-monitor capture
via ``SM_CXVIRTUALSCREEN`` / ``EnumDisplayMonitors`` is a future
``Screenshot(display=N)`` extension.

**DPI.** :mod:`fastgrab.backends.base` promises device pixels, and Win32
only tells the truth to a DPI-aware caller: ``GetSystemMetrics`` is
virtualized to 96 DPI for a DPI-unaware thread, and ``BitBlt`` against
the desktop reads the virtualized (stretched) surface, so at a 150%
display scale both would speak logical units. CPython's manifest
declares no awareness, so an unpatched process inherits "unaware" and
what fastgrab reported would depend on whatever host application
happened to embed it.

The fix is :func:`_thread_dpi_aware`: per-monitor-v2 awareness is set on
the **calling thread only**, for the duration of the metrics query, the
screen-DC acquisition and the blit, then restored. Process-wide
awareness (``SetProcessDpiAwarenessContext``) is deliberately not used —
it is a one-shot, irreversible property that would silently re-scale the
UI of any application that merely imported fastgrab.

Byte order matches the rest of fastgrab: 32-bpp ``BITMAPINFOHEADER`` on
little-endian Windows lays each pixel out as B, G, R, A in memory, so
the captured numpy array is BGRA — same contract as X11 and wlr.
"""
from __future__ import annotations

import contextlib
import ctypes
from ctypes import wintypes  # only available on Windows; gated by importer

from .base import BaseBackend


# Constants from <wingdi.h> / <winuser.h>
_SM_CXSCREEN = 0
_SM_CYSCREEN = 1
_SRCCOPY = 0x00CC0020
_CAPTUREBLT = 0x40000000
_BI_RGB = 0
_DIB_RGB_COLORS = 0

# DPI_AWARENESS_CONTEXT is an opaque HANDLE whose predefined values are
# small negative integers cast to a pointer (<winuser.h>). -4 is
# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, the only context that
# reports true physical pixels on every monitor of a mixed-DPI desktop.
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = wintypes.HANDLE(-4)


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        # bmiColors is a flexible array; for 32-bpp BI_RGB we don't use it.
        ("bmiColors", wintypes.DWORD * 3),
    ]


def _load_libs():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int

    # Windows 10 1607+. Older systems simply do not export it, and
    # ctypes resolves exports lazily through __getattr__ — so probe with
    # getattr rather than calling it and catching the failure, which on a
    # partially-applied call could leave a changed context on the thread.
    set_dpi_ctx = getattr(user32, "SetThreadDpiAwarenessContext", None)
    if set_dpi_ctx is not None:
        set_dpi_ctx.argtypes = [wintypes.HANDLE]
        # Returns the thread's PREVIOUS context, or NULL if the value
        # passed in was invalid. That return is the whole "save" half of
        # save/restore, which is why GetThreadDpiAwarenessContext is not
        # wired up here.
        set_dpi_ctx.restype = wintypes.HANDLE

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.CreateDIBSection.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(_BITMAPINFO),
        wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.BitBlt.argtypes = [
        wintypes.HDC, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        wintypes.HDC, ctypes.c_int, ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL

    return user32, gdi32


@contextlib.contextmanager
def _thread_dpi_aware(user32):
    """Make the calling thread per-monitor-v2 DPI aware for the block.

    Scoped to this thread, and restored on the way out even if the body
    raises — fastgrab is a library, and leaving a changed DPI context
    behind on a thread it does not own would re-scale the caller's own
    windows.

    Degrades to a no-op on Windows earlier than 10 1607, where
    ``SetThreadDpiAwarenessContext`` does not exist, and on the
    documented NULL return that says the context value was rejected. In
    both cases nothing was changed, so there is nothing to restore and
    the caller simply gets the host process's context — the pre-fix
    behaviour, rather than a crash.

    Yields whether awareness was actually established.
    """
    set_dpi_ctx = getattr(user32, "SetThreadDpiAwarenessContext", None)
    previous = None
    if set_dpi_ctx is not None:
        previous = set_dpi_ctx(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    try:
        yield bool(previous)
    finally:
        if previous:
            set_dpi_ctx(previous)


class WindowsBackend(BaseBackend):
    def __init__(self):
        self._user32, self._gdi32 = _load_libs()

        # The memory DC is built once and kept: CreateCompatibleDC copies
        # the device's characteristics, so the DC it returns outlives the
        # screen DC it was derived from and carries no DPI context of its
        # own (DPI virtualization applies to on-screen surfaces, not to
        # memory bitmaps).
        with _thread_dpi_aware(self._user32):
            screen_dc = self._user32.GetDC(None)
            if not screen_dc:
                raise RuntimeError("GetDC(NULL) returned 0; cannot access screen")
            try:
                self._mem_dc = self._gdi32.CreateCompatibleDC(screen_dc)
            finally:
                self._user32.ReleaseDC(None, screen_dc)
        if not self._mem_dc:
            raise RuntimeError("CreateCompatibleDC failed")

        # cached DIBSection state — reused when (w, h) match the prior call.
        self._cur_w = 0
        self._cur_h = 0
        self._cur_bitmap = None     # HBITMAP
        self._cur_old_obj = None    # HGDIOBJ returned by SelectObject
        self._cur_bits_ptr = None   # raw pointer into the DIBSection's pixels

    # -------- BaseBackend API --------

    def resolution(self):
        # SM_CXSCREEN/SM_CYSCREEN, not the SM_CXVIRTUALSCREEN family:
        # this backend blits from the desktop DC, whose origin is the
        # primary monitor's top-left, so reporting the virtual screen
        # would hand the caller a bbox space that does not match the one
        # screenshot() indexes — the virtual screen extends into negative
        # coordinates when a monitor sits left of or above the primary.
        # Reconciling the two is the Screenshot(display=N) feature, not
        # this one. Under the per-monitor-v2 context these are the
        # primary monitor's physical pixels; outside it they would be
        # 96-DPI logical units.
        with _thread_dpi_aware(self._user32):
            w = self._user32.GetSystemMetrics(_SM_CXSCREEN)
            h = self._user32.GetSystemMetrics(_SM_CYSCREEN)
        return (int(w), int(h))

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        h, w, _ = img.shape
        self._ensure_bitmap(w, h)

        # The screen DC is acquired inside the awareness scope and let go
        # again on the way out rather than cached across calls: a DC
        # handed to a DPI-unaware thread keeps reading the virtualized
        # desktop regardless of which context is in force when BitBlt
        # finally runs, so a DC cached at construction time would quietly
        # undo this fix for anybody who built the backend from an unaware
        # thread. The caching that makes this backend fast is the
        # DIBSection and its memory DC, and both are untouched; a
        # GetDC/ReleaseDC pair costs microseconds against a blit measured
        # in milliseconds. It also stops fastgrab from sitting on a
        # common-cache DC for the life of the process.
        with _thread_dpi_aware(self._user32):
            screen_dc = self._user32.GetDC(None)
            if not screen_dc:
                raise RuntimeError("GetDC(NULL) returned 0; cannot access screen")
            try:
                ok = self._gdi32.BitBlt(
                    self._mem_dc, 0, 0, w, h,
                    screen_dc, int(x), int(y),
                    _SRCCOPY | _CAPTUREBLT,
                )
                # Read before ReleaseDC: that call goes through the same
                # use_last_error=True handle and would overwrite it.
                err = 0 if ok else ctypes.get_last_error()
            finally:
                self._user32.ReleaseDC(None, screen_dc)
        if not ok:
            raise RuntimeError("BitBlt failed (GetLastError={})".format(err))

        nbytes = w * h * 4
        ctypes.memmove(img.ctypes.data, self._cur_bits_ptr, nbytes)

    # -------- DIBSection cache --------

    def _ensure_bitmap(self, w, h):
        if self._cur_bitmap is not None and (w, h) == (self._cur_w, self._cur_h):
            return

        # Tear down previous bitmap.
        if self._cur_bitmap is not None:
            if self._cur_old_obj is not None:
                self._gdi32.SelectObject(self._mem_dc, self._cur_old_obj)
            self._gdi32.DeleteObject(self._cur_bitmap)
            self._cur_bitmap = None
            self._cur_old_obj = None
            self._cur_bits_ptr = None

        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        # Negative height → top-down DIB so memory rows match numpy
        # row-major order; no flipud needed.
        bmi.bmiHeader.biHeight = -h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = _BI_RGB
        bmi.bmiHeader.biSizeImage = w * h * 4

        bits_ptr = ctypes.c_void_p()
        bitmap = self._gdi32.CreateDIBSection(
            self._mem_dc,
            ctypes.byref(bmi),
            _DIB_RGB_COLORS,
            ctypes.byref(bits_ptr),
            None,
            0,
        )
        if not bitmap or not bits_ptr.value:
            raise RuntimeError(
                "CreateDIBSection failed for {}x{} (GetLastError={})".format(
                    w, h, ctypes.get_last_error()
                )
            )

        old_obj = self._gdi32.SelectObject(self._mem_dc, bitmap)
        if not old_obj:
            self._gdi32.DeleteObject(bitmap)
            raise RuntimeError("SelectObject failed for new DIBSection")

        self._cur_bitmap = bitmap
        self._cur_old_obj = old_obj
        self._cur_bits_ptr = bits_ptr.value
        self._cur_w = w
        self._cur_h = h

    # No __del__: matching the wlr backend, we let the OS reclaim the
    # memory DC and bitmap handles at process exit. Explicit cleanup
    # paths are race-prone under interpreter shutdown. The screen DC is
    # no longer among them — it is released on every capture.
