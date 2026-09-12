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
import weakref
from ctypes import wintypes  # only available on Windows; gated by importer

import numpy

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


def _validate_destination(img):
    """Reject a destination this backend cannot safely fill.

    ``screenshot()`` copies with :func:`ctypes.memmove`, which walks
    ``height * width * 4`` bytes forward from the array's data pointer
    and knows nothing about strides. A non-contiguous view satisfies the
    documented shape and dtype while pointing somewhere else entirely --
    ``arr[::-1]`` starts at the *last* row -- so the copy would run past
    the end of the allocation and corrupt the heap.

    This backend is alone in needing the guard: macOS and wlr assign
    through numpy (``img[:] = ...``), which is stride-aware and refuses a
    read-only destination by itself. The checks and their wording mirror
    the X11 C extension's, so the directly-callable backend API rejects
    the same things on every platform. Rejected rather than coerced: a
    coerced copy would be filled and then dropped on the floor.
    """
    if not isinstance(img, numpy.ndarray):
        raise TypeError("image buffer must be a numpy ndarray")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("image buffer must be a (height, width, 4) array")
    if img.dtype != numpy.uint8:
        raise ValueError("image buffer must have dtype uint8")
    flags = img.flags
    if not (flags["C_CONTIGUOUS"] and flags["ALIGNED"] and flags["WRITEABLE"]):
        raise ValueError(
            "image buffer must be C-contiguous, aligned and writable"
        )
    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image buffer height and width must be positive")


def _release_gdi(state):
    """Delete a memory DC and the DIBSection selected into it.

    Order is forced by GDI: a bitmap that is still selected into a DC
    cannot be deleted, so the DC's original bitmap goes back first, and
    the DC itself is deleted last because it is what the bitmap was
    selected into.

    ``state`` is a :class:`_GdiSurface`'s attribute dict — see that class
    for why the handles arrive by dict rather than by value. Nothing here
    raises: it runs from a finalizer, and these calls only report failure
    through a return code anyway.
    """
    gdi32 = state.get("gdi32")
    if gdi32 is None:
        return
    mem_dc = state.get("mem_dc")
    bitmap = state.get("bitmap")
    old_obj = state.get("old_obj")
    if bitmap:
        if old_obj:
            gdi32.SelectObject(mem_dc, old_obj)
        gdi32.DeleteObject(bitmap)
    if mem_dc:
        gdi32.DeleteDC(mem_dc)
    state["bitmap"] = None
    state["old_obj"] = None
    state["bits_ptr"] = None
    state["mem_dc"] = None


class _GdiSurface:
    """Owns one backend's memory DC and its current DIBSection.

    Freeing both when this object is dropped is what bounds the
    backend's GDI handles to its own lifetime without putting a
    ``__del__`` on the backend. GDI's per-process quota is 10,000
    handles, but the DIBSection's pixels leak along with the handle — at
    1080p that is ~8 MB apiece, so a program that builds a Screenshot per
    frame runs out of memory long before it runs out of handles.

    The finalizer is handed this instance's ``__dict__`` rather than the
    handles themselves. ``weakref.finalize`` freezes its arguments at
    registration time, and ``bitmap`` is replaced every time the capture
    size changes, so passing the handle would free a stale one and leak
    the live one. The attribute dict outlives the instance and always
    reads back what is currently open. It must not be handed ``self`` —
    a finalizer that references its own object can never run.

    ``atexit`` is off: at interpreter shutdown Windows reclaims every
    handle the process owns, and running GDI calls that late is exactly
    the race the backend's old "no ``__del__``" note was avoiding.
    """

    def __init__(self, gdi32, mem_dc):
        self.gdi32 = gdi32
        self.mem_dc = mem_dc
        self.bitmap = None      # HBITMAP
        self.old_obj = None     # HGDIOBJ displaced by SelectObject
        self.bits_ptr = None    # raw pointer into the DIBSection's pixels
        self.w = 0
        self.h = 0
        self._finalize = weakref.finalize(self, _release_gdi, self.__dict__)
        self._finalize.atexit = False

    def close(self):
        """Free now. Idempotent — a finalizer only ever fires once."""
        self._finalize()


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
                mem_dc = self._gdi32.CreateCompatibleDC(screen_dc)
            finally:
                self._user32.ReleaseDC(None, screen_dc)
        if not mem_dc:
            raise RuntimeError("CreateCompatibleDC failed")

        # The DC and the DIBSection cached against it are held by a
        # separate owner so that dropping this backend frees them; see
        # _GdiSurface. The bitmap is reused whenever (w, h) match the
        # previous call.
        self._surface = _GdiSurface(self._gdi32, mem_dc)

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
        # Before anything touches the screen: the copy at the end is a
        # raw memmove and cannot recover from a bad destination.
        self._check_open()
        _validate_destination(img)
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
                    self._surface.mem_dc, 0, 0, w, h,
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
        ctypes.memmove(img.ctypes.data, self._surface.bits_ptr, nbytes)

    # -------- DIBSection cache --------

    def _ensure_bitmap(self, w, h):
        surface = self._surface
        if surface.bitmap is not None and (w, h) == (surface.w, surface.h):
            return

        # Tear down previous bitmap.
        if surface.bitmap is not None:
            if surface.old_obj is not None:
                self._gdi32.SelectObject(surface.mem_dc, surface.old_obj)
            self._gdi32.DeleteObject(surface.bitmap)
            surface.bitmap = None
            surface.old_obj = None
            surface.bits_ptr = None

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
            surface.mem_dc,
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

        old_obj = self._gdi32.SelectObject(surface.mem_dc, bitmap)
        if not old_obj:
            self._gdi32.DeleteObject(bitmap)
            raise RuntimeError("SelectObject failed for new DIBSection")

        surface.bitmap = bitmap
        surface.old_obj = old_obj
        surface.bits_ptr = bits_ptr.value
        surface.w = w
        surface.h = h

    def close(self):
        """Release the memory DC and the DIBSection selected into it.

        Optional — dropping the backend frees the same handles through
        :class:`_GdiSurface`'s finalizer. Call it to pick the moment.

        Still no ``__del__`` on the backend: explicit cleanup driven from
        interpreter shutdown is the race the old note here warned about,
        and the finalizer deliberately does not run then. The screen DC
        was never among these handles — it is released on every capture.
        """
        self._surface.close()
        self._closed = True
