"""Windows screen capture backend (BitBlt + DIBSection via ctypes).

Pure-Python: no extras, no ``pywin32``. We talk to ``user32.dll`` and
``gdi32.dll`` directly through :mod:`ctypes` so a default
``pip install fastgrab`` works on a fresh Windows 10 / 11 install with
nothing but Python + numpy.

V1 captures from the **primary** monitor only. Multi-monitor capture
via ``SM_CXVIRTUALSCREEN`` / ``EnumDisplayMonitors`` is a future
``Screenshot(display=N)`` extension.

Byte order matches the rest of fastgrab: 32-bpp ``BITMAPINFOHEADER`` on
little-endian Windows lays each pixel out as B, G, R, A in memory, so
the captured numpy array is BGRA — same contract as X11 and wlr.
"""
from __future__ import annotations

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


class WindowsBackend(BaseBackend):
    def __init__(self):
        self._user32, self._gdi32 = _load_libs()
        self._screen_dc = self._user32.GetDC(None)
        if not self._screen_dc:
            raise RuntimeError("GetDC(NULL) returned 0; cannot access screen")
        self._mem_dc = self._gdi32.CreateCompatibleDC(self._screen_dc)
        if not self._mem_dc:
            self._user32.ReleaseDC(None, self._screen_dc)
            raise RuntimeError("CreateCompatibleDC failed")

        # cached DIBSection state — reused when (w, h) match the prior call.
        self._cur_w = 0
        self._cur_h = 0
        self._cur_bitmap = None     # HBITMAP
        self._cur_old_obj = None    # HGDIOBJ returned by SelectObject
        self._cur_bits_ptr = None   # raw pointer into the DIBSection's pixels

    # -------- BaseBackend API --------

    def resolution(self):
        w = self._user32.GetSystemMetrics(_SM_CXSCREEN)
        h = self._user32.GetSystemMetrics(_SM_CYSCREEN)
        return (int(w), int(h))

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        h, w, _ = img.shape
        self._ensure_bitmap(w, h)

        ok = self._gdi32.BitBlt(
            self._mem_dc, 0, 0, w, h,
            self._screen_dc, int(x), int(y),
            _SRCCOPY | _CAPTUREBLT,
        )
        if not ok:
            err = ctypes.get_last_error()
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

    # No __del__: matching the wlr backend, we let the OS reclaim DC and
    # bitmap handles at process exit. Explicit cleanup paths are race-prone
    # under interpreter shutdown.
