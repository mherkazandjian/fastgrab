"""Tests for the Windows backend: the DPI-awareness scope around Win32.

Win32 only reports device pixels to a DPI-aware caller, so the backend
temporarily gives its own thread per-monitor-v2 awareness around the
metrics query, the screen-DC acquisition and the blit. None of that can
run on the Linux CI container, so ``user32``/``gdi32`` are faked and
``ctypes.wintypes`` is stubbed in (it refuses to import off Windows).

The fakes model Win32 where it is *unforgiving*, the way the macOS
backend's fakes do:

* the thread's DPI context is real state — ``GetSystemMetrics`` returns
  96-DPI-virtualized numbers whenever the thread is unaware, so a
  missing or leaked context switch shows up as wrong values rather than
  only as an un-asserted call log;
* a screen DC remembers the awareness it was acquired under, and
  ``BitBlt`` serves the virtualized desktop for a DC acquired by an
  unaware thread no matter what context is in force at blit time —
  which is exactly why the cached screen DC had to go, and which a call
  log cannot see at all;
* ``ReleaseDC`` clobbers the shared last-error value the way a second
  call through a ``use_last_error=True`` handle does, so reading
  ``GetLastError`` in the wrong order reports the wrong code;
* an out-of-range blit raises instead of letting numpy clip it into a
  short, silently zero-padded region;
* ``SetThreadDpiAwarenessContext`` is bound in ``__init__`` rather than
  defined on the class, so ``dpi_api=False`` models an older Windows by
  genuinely not having the export — a bare call would raise
  ``AttributeError`` here just as it would there.
"""
import ctypes
import sys
import types

import numpy
import pytest


def _stub_wintypes():
    """The ``ctypes.wintypes`` names the backend needs, off Windows.

    ``import ctypes.wintypes`` raises on non-Windows (``VARIANT_BOOL``'s
    ``_type_ = "v"`` is unsupported there), which would take this whole
    module down at collection time. The aliases below are the real
    definitions verbatim, so the structure layouts under test are the
    ones Windows would build.
    """
    stub = types.ModuleType("ctypes.wintypes")
    stub.BYTE = ctypes.c_byte
    stub.WORD = ctypes.c_ushort
    stub.DWORD = ctypes.c_ulong
    stub.LONG = ctypes.c_long
    stub.UINT = ctypes.c_uint
    stub.BOOL = ctypes.c_long
    stub.HANDLE = ctypes.c_void_p
    for alias in ("HWND", "HDC", "HBITMAP", "HGDIOBJ"):
        setattr(stub, alias, stub.HANDLE)
    return stub


try:  # pragma: no cover - one branch per platform
    from ctypes import wintypes as _wintypes  # noqa: F401
except (ImportError, ValueError):
    _stub = _stub_wintypes()
    sys.modules.setdefault("ctypes.wintypes", _stub)
    ctypes.wintypes = _stub

from fastgrab.backends import windows  # noqa: E402
from fastgrab.backends.windows import (  # noqa: E402
    WindowsBackend,
    _BITMAPINFO,
    _DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
    _SM_CXSCREEN,
    _SM_CYSCREEN,
    _thread_dpi_aware,
)

# Fakes all the way down — no Win32, no display server, any platform.
pytestmark = pytest.mark.no_display


def _handle_value(context):
    """The integer a DPI_AWARENESS_CONTEXT handle carries."""
    return getattr(context, "value", context)


_PER_MONITOR_AWARE_V2 = _handle_value(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
# DPI_AWARENESS_CONTEXT_UNAWARE — what a stock CPython process inherits,
# and what the backend must hand back when it is done.
_HOST_CONTEXT = ctypes.c_void_p(-1).value

# A 64x48-pixel panel driven at 200%, so a DPI-unaware caller sees 32x24.
PHYSICAL = (64, 48)
SCALE = 2

_DEFAULT_BITMAP = 0x4000
_BITBLT_ERROR = 5

# Stands in for the thread's Win32 last-error slot, which every call
# through a use_last_error=True handle overwrites.
_LAST_ERROR = {"code": 0}


class _FakeUser32:
    """The user32 surface the backend touches, over a real DPI context."""

    def __init__(self, physical=PHYSICAL, scale=SCALE, dpi_api=True,
                 get_dc_returns=None):
        self.physical_w, self.physical_h = physical
        self.scale = scale
        self.screen = numpy.random.default_rng(0).integers(
            0, 256, (self.physical_h, self.physical_w, 4), dtype=numpy.uint8)
        # What a DPI-unaware caller is shown: the desktop stretched down
        # to 96 DPI. Nearest-neighbour is enough to make it distinguishable
        # from the real backing store.
        self.virtual_screen = self.screen[::scale, ::scale]

        self.thread_context = _HOST_CONTEXT
        self.context_log = []

        self.get_dc_returns = get_dc_returns
        self._dc_serial = 0
        self.dc_awareness = {}   # every hdc ever handed out
        self.live_dcs = set()    # the ones not released yet
        self.released_dcs = []

        if dpi_api:
            # Bound here, not defined on the class: dpi_api=False must
            # look like an export that is simply not there.
            self.SetThreadDpiAwarenessContext = self._set_thread_dpi

    # -------- DPI awareness --------

    def _set_thread_dpi(self, context):
        self.context_log.append(_handle_value(context))
        previous, self.thread_context = self.thread_context, context
        return previous

    def aware(self):
        return _handle_value(self.thread_context) == _PER_MONITOR_AWARE_V2

    # -------- metrics --------

    def GetSystemMetrics(self, index):
        w, h = self.physical_w, self.physical_h
        if not self.aware():
            # Windows virtualizes screen metrics for a DPI-unaware thread.
            w, h = w // self.scale, h // self.scale
        return {_SM_CXSCREEN: w, _SM_CYSCREEN: h}[index]

    # -------- device contexts --------

    def GetDC(self, hwnd):
        assert hwnd is None, "the backend only ever asks for the screen DC"
        if self.get_dc_returns is not None:
            return self.get_dc_returns
        self._dc_serial += 1
        hdc = 0x1000 + self._dc_serial
        # Latched at acquisition: this is the property a cached DC would
        # carry forward across a later context switch.
        self.dc_awareness[hdc] = self.aware()
        self.live_dcs.add(hdc)
        return hdc

    def ReleaseDC(self, hwnd, hdc):
        assert hdc in self.live_dcs, \
            "released a DC that was never acquired (or released twice)"
        self.live_dcs.discard(hdc)
        self.released_dcs.append(hdc)
        _LAST_ERROR["code"] = 0
        return 1


class _FakeGdi32:
    """The gdi32 surface, blitting out of the user32 fake's screen."""

    def __init__(self, user32):
        self._user32 = user32
        self.blits = []
        self.bitblt_returns = 1
        self.selected = _DEFAULT_BITMAP
        self.buffers = {}         # HBITMAP -> (ctypes buffer, w, h)
        self.created_bitmaps = []
        self.deleted_bitmaps = []
        self._bitmap_serial = 0

    def CreateCompatibleDC(self, hdc):
        assert hdc, "a memory DC must be built from a live screen DC"
        return 0x2000

    def CreateDIBSection(self, hdc, bmi, usage, bits_ptr, section, offset):
        # ctypes.byref() wraps the argument in a CArgObject; the fake
        # reads the original back out rather than pretending to be C.
        header = getattr(bmi, "_obj", bmi).bmiHeader
        out = getattr(bits_ptr, "_obj", bits_ptr)
        assert header.biHeight < 0, "the DIB must stay top-down"
        w, h = header.biWidth, -header.biHeight
        buf = (ctypes.c_ubyte * (w * h * 4))()
        self._bitmap_serial += 1
        bitmap = 0x3000 + self._bitmap_serial
        self.buffers[bitmap] = (buf, w, h)
        self.created_bitmaps.append(bitmap)
        out.value = ctypes.addressof(buf)
        return bitmap

    def SelectObject(self, hdc, hgdiobj):
        previous, self.selected = self.selected, hgdiobj
        return previous

    def DeleteObject(self, hgdiobj):
        self.deleted_bitmaps.append(hgdiobj)
        return 1

    def BitBlt(self, dst, dx, dy, w, h, src, sx, sy, rop):
        src_aware = self._user32.dc_awareness[src]
        self.blits.append(dict(dst=dst, dx=dx, dy=dy, w=w, h=h, src=src,
                               sx=sx, sy=sy, rop=rop, src_aware=src_aware,
                               thread_aware=self._user32.aware()))
        if not self.bitblt_returns:
            _LAST_ERROR["code"] = _BITBLT_ERROR
            return 0
        # A DC acquired by a DPI-unaware thread keeps reading the
        # virtualized desktop however aware the thread has since become.
        source = (self._user32.screen if src_aware
                  else self._user32.virtual_screen)
        max_h, max_w = source.shape[:2]
        if sx < 0 or sy < 0 or sx + w > max_w or sy + h > max_h:
            # numpy would clip this to a short region and leave the rest
            # of the DIB zeroed — a silently "valid" capture.
            raise AssertionError(
                "blit of %r falls outside the %dx%d surface this DC sees"
                % ((sx, sy, w, h), max_w, max_h))
        buf, buf_w, buf_h = self.buffers[self.selected]
        assert (buf_w, buf_h) == (w, h), "the blit does not fill the DIBSection"
        region = numpy.ascontiguousarray(source[sy:sy + h, sx:sx + w])
        ctypes.memmove(buf, region.ctypes.data, region.nbytes)
        return 1


@pytest.fixture(autouse=True)
def _last_error(monkeypatch):
    """``ctypes.get_last_error`` is Windows-only; the error paths call it."""
    _LAST_ERROR["code"] = 0
    monkeypatch.setattr(ctypes, "get_last_error",
                        lambda: _LAST_ERROR["code"], raising=False)


@pytest.fixture
def make_backend(monkeypatch):
    """Build a WindowsBackend over fakes, running its real ``__init__``."""
    made = []

    def _make(**kwargs):
        user32 = _FakeUser32(**kwargs)
        gdi32 = _FakeGdi32(user32)
        made.append((user32, gdi32))
        monkeypatch.setattr(windows, "_load_libs", lambda: (user32, gdi32))
        return WindowsBackend(), user32, gdi32

    _make.fakes = made
    return _make


def _capture(backend, x, y, w, h):
    img = numpy.zeros((h, w, 4), numpy.uint8)
    backend.screenshot(x, y, img)
    return img


def _bmi(w, h):
    bmi = _BITMAPINFO()
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    return ctypes.byref(bmi)


# -------- resolution() --------

def test_resolution_reports_physical_pixels_at_a_200_percent_scale(make_backend):
    """The bug: an unaware thread is shown 32x24 for a 64x48 panel."""
    backend, user32, _ = make_backend()
    # queried outside the backend's scope, i.e. what it used to see
    assert user32.GetSystemMetrics(_SM_CXSCREEN) == 32
    assert backend.resolution() == PHYSICAL


def test_resolution_sets_and_restores_the_thread_dpi_context(make_backend):
    backend, user32, _ = make_backend()
    before = len(user32.context_log)
    assert backend.resolution() == PHYSICAL
    assert user32.context_log[before:] == [_PER_MONITOR_AWARE_V2, _HOST_CONTEXT]
    assert user32.thread_context == _HOST_CONTEXT


def test_resolution_restores_the_thread_dpi_context_when_it_raises(make_backend):
    """A library must not leave a changed DPI context on the caller."""
    backend, user32, _ = make_backend()

    def boom(index):
        raise OSError("GetSystemMetrics failed")

    user32.GetSystemMetrics = boom
    with pytest.raises(OSError):
        backend.resolution()
    assert user32.thread_context == _HOST_CONTEXT


def test_resolution_falls_back_when_the_dpi_api_is_missing(make_backend):
    """Windows before 10 1607: no export, so the host context stands."""
    backend, user32, _ = make_backend(dpi_api=False)
    assert user32.context_log == []
    assert backend.resolution() == (32, 24)   # virtualized, as before the fix


# -------- screenshot() --------

def test_screenshot_sets_and_restores_the_thread_dpi_context(make_backend):
    backend, user32, _ = make_backend()
    before = len(user32.context_log)
    _capture(backend, 0, 0, 8, 6)
    assert user32.context_log[before:] == [_PER_MONITOR_AWARE_V2, _HOST_CONTEXT]
    assert user32.thread_context == _HOST_CONTEXT


def test_screenshot_restores_the_thread_dpi_context_when_the_blit_fails(
        make_backend):
    backend, user32, gdi32 = make_backend()
    gdi32.bitblt_returns = 0
    with pytest.raises(RuntimeError, match="BitBlt failed"):
        _capture(backend, 0, 0, 8, 6)
    assert user32.thread_context == _HOST_CONTEXT
    assert user32.live_dcs == set(), "the screen DC leaked on the error path"


def test_screenshot_restores_the_thread_dpi_context_when_getdc_fails(
        make_backend):
    backend, user32, _ = make_backend()
    user32.get_dc_returns = 0
    with pytest.raises(RuntimeError, match="GetDC"):
        _capture(backend, 0, 0, 8, 6)
    assert user32.thread_context == _HOST_CONTEXT


def test_blit_error_reports_the_code_from_before_releasedc(make_backend):
    """ReleaseDC runs through the same use_last_error handle and clobbers it."""
    backend, _, gdi32 = make_backend()
    gdi32.bitblt_returns = 0
    with pytest.raises(RuntimeError, match=r"GetLastError=5\b"):
        _capture(backend, 0, 0, 8, 6)


def test_screenshot_falls_back_when_the_dpi_api_is_missing(make_backend):
    """No export means no crash — the pre-fix behaviour, not an error."""
    backend, user32, _ = make_backend(dpi_api=False)
    img = _capture(backend, 3, 2, 8, 6)
    assert user32.context_log == []
    assert numpy.array_equal(img, user32.virtual_screen[2:8, 3:11])


# -------- the screen DC --------

def test_the_screen_dc_is_acquired_inside_the_awareness_scope(make_backend):
    """A DC acquired before the switch would blit the virtualized desktop."""
    backend, _, gdi32 = make_backend()
    _capture(backend, 0, 0, 8, 6)
    blit, = gdi32.blits
    assert blit["src_aware"] is True
    assert blit["thread_aware"] is True


def test_construction_acquires_its_screen_dc_inside_the_scope(make_backend):
    _, user32, _ = make_backend()
    init_dc, = list(user32.dc_awareness)
    assert user32.dc_awareness[init_dc] is True


def test_construction_does_not_retain_a_screen_dc(make_backend):
    """The memory DC outlives the screen DC it was made compatible with."""
    backend, user32, _ = make_backend()
    assert user32.live_dcs == set()
    assert len(user32.released_dcs) == 1
    assert not hasattr(backend, "_screen_dc")


def test_every_capture_acquires_and_releases_its_own_screen_dc(make_backend):
    backend, user32, _ = make_backend()
    for _ in range(3):
        _capture(backend, 0, 0, 8, 6)
    # one from __init__ plus one per capture, all handed back
    assert len(user32.released_dcs) == 4
    assert len(set(user32.released_dcs)) == 4
    assert user32.live_dcs == set()


# -------- coordinates --------

_REGIONS = [
    (0, 0, 64, 48),      # the whole physical panel
    (0, 0, 8, 6),
    (7, 5, 11, 9),       # odd offset and size
    (33, 25, 4, 4),      # past the 32x24 a DPI-unaware caller could reach
    (63, 47, 1, 1),      # the bottom-right physical pixel
]


@pytest.mark.parametrize("x, y, w, h", _REGIONS)
def test_coordinates_reach_bitblt_as_device_pixels(make_backend, x, y, w, h):
    """No scale factor anywhere: the region is passed through untouched."""
    backend, _, gdi32 = make_backend()
    _capture(backend, x, y, w, h)
    blit, = gdi32.blits
    assert (blit["sx"], blit["sy"]) == (x, y)
    assert (blit["w"], blit["h"]) == (w, h)
    assert (blit["dx"], blit["dy"]) == (0, 0)


@pytest.mark.parametrize("x, y, w, h", _REGIONS)
def test_capture_matches_the_physical_backing_store(make_backend, x, y, w, h):
    backend, user32, _ = make_backend()
    img = _capture(backend, x, y, w, h)
    assert img.shape == (h, w, 4)
    assert numpy.array_equal(img, user32.screen[y:y + h, x:x + w])


def test_capture_is_unchanged_at_a_100_percent_scale(make_backend):
    """The case CI and most desktops see; the fix must be a no-op there."""
    backend, user32, _ = make_backend(scale=1)
    assert backend.resolution() == PHYSICAL
    assert numpy.array_equal(_capture(backend, 7, 5, 11, 9),
                             user32.screen[5:14, 7:18])


# -------- the DIBSection cache is not collateral damage --------

def test_the_dibsection_is_reused_for_a_repeated_size(make_backend):
    backend, _, gdi32 = make_backend()
    for _ in range(3):
        _capture(backend, 0, 0, 8, 6)
    assert len(gdi32.created_bitmaps) == 1
    assert gdi32.deleted_bitmaps == []


def test_a_new_size_replaces_the_dibsection(make_backend):
    backend, _, gdi32 = make_backend()
    _capture(backend, 0, 0, 8, 6)
    _capture(backend, 0, 0, 12, 10)
    assert len(gdi32.created_bitmaps) == 2
    assert gdi32.deleted_bitmaps == gdi32.created_bitmaps[:1]


# -------- the context manager itself --------

def test_thread_dpi_aware_reports_whether_awareness_took():
    user32 = _FakeUser32()
    with _thread_dpi_aware(user32) as aware:
        assert aware is True
        assert user32.aware()
    assert user32.thread_context == _HOST_CONTEXT


def test_thread_dpi_aware_is_a_no_op_without_the_export():
    user32 = _FakeUser32(dpi_api=False)
    with _thread_dpi_aware(user32) as aware:
        assert aware is False
    assert user32.thread_context == _HOST_CONTEXT
    assert user32.context_log == []


def test_thread_dpi_aware_does_not_restore_a_rejected_context():
    """NULL means nothing was changed, so there is nothing to put back."""
    user32 = _FakeUser32()
    user32.SetThreadDpiAwarenessContext = lambda context: None
    with _thread_dpi_aware(user32) as aware:
        assert aware is False
    assert user32.thread_context == _HOST_CONTEXT


def test_thread_dpi_aware_restores_on_an_exception():
    user32 = _FakeUser32()
    with pytest.raises(ZeroDivisionError):
        with _thread_dpi_aware(user32):
            1 / 0
    assert user32.thread_context == _HOST_CONTEXT


# -------- the fakes themselves --------

def test_an_out_of_range_blit_is_not_silently_zero_padded():
    """Guards the fake: numpy slicing would have clipped these away."""
    user32 = _FakeUser32()
    gdi32 = _FakeGdi32(user32)
    hdc = user32.GetDC(None)
    gdi32.CreateDIBSection(0x2000, _bmi(10, 10), 0,
                           ctypes.byref(ctypes.c_void_p()), None, 0)
    gdi32.selected = gdi32.created_bitmaps[-1]
    with pytest.raises(AssertionError, match="outside the"):
        gdi32.BitBlt(0x2000, 0, 0, 10, 10, hdc, 60, 44, 0)


def test_a_stale_screen_dc_would_serve_the_virtualized_desktop():
    """Guards the fake that makes the cached-DC regression detectable."""
    user32 = _FakeUser32()
    gdi32 = _FakeGdi32(user32)
    stale = user32.GetDC(None)               # acquired while unaware
    with _thread_dpi_aware(user32):
        assert user32.aware()
        gdi32.CreateDIBSection(0x2000, _bmi(8, 6), 0,
                               ctypes.byref(ctypes.c_void_p()), None, 0)
        gdi32.selected = gdi32.created_bitmaps[-1]
        gdi32.BitBlt(0x2000, 0, 0, 8, 6, stale, 0, 0, 0)
    buf, _, _ = gdi32.buffers[gdi32.selected]
    got = numpy.frombuffer(buf, numpy.uint8).reshape(6, 8, 4)
    assert numpy.array_equal(got, user32.virtual_screen[0:6, 0:8])
    assert not numpy.array_equal(got, user32.screen[0:6, 0:8])


# -------- destination validation --------
#
# screenshot() finishes with a raw ctypes.memmove, which walks
# height*width*4 bytes forward from the array's data pointer and knows
# nothing about strides. Everything below would previously have reached
# that memmove. The reversed-view case is the dangerous one: it satisfies
# the documented shape and dtype while its data pointer sits at the *last*
# row, so the copy ran off the end of the allocation.


def _reject(backend, gdi32, img, exc, match):
    with pytest.raises(exc, match=match):
        backend.screenshot(0, 0, img)
    # Refused before anything touched the screen.
    assert gdi32.blits == []


def test_reversed_view_is_rejected_not_overrun(make_backend):
    backend, _user32, gdi32 = make_backend()
    buf = numpy.zeros((16, 16, 4), numpy.uint8)
    view = buf[::-1]
    # Sanity: the case under test. Same shape and dtype, data pointer at
    # the far end of the allocation.
    assert view.shape == buf.shape and view.dtype == buf.dtype
    assert not view.flags["C_CONTIGUOUS"]
    assert view.ctypes.data > buf.ctypes.data
    _reject(backend, gdi32, view, ValueError, "C-contiguous")


def test_strided_view_is_rejected(make_backend):
    backend, _user32, gdi32 = make_backend()
    view = numpy.zeros((8, 32, 4), numpy.uint8)[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    _reject(backend, gdi32, view, ValueError, "C-contiguous")


def test_read_only_buffer_is_rejected(make_backend):
    backend, _user32, gdi32 = make_backend()
    buf = numpy.zeros((8, 8, 4), numpy.uint8)
    buf.flags.writeable = False
    _reject(backend, gdi32, buf, ValueError, "writable")


def test_wrong_channel_count_is_rejected(make_backend):
    # Regression shape: (8, 8, 3) is 192 bytes but the copy writes 256.
    backend, _user32, gdi32 = make_backend()
    _reject(backend, gdi32, numpy.zeros((8, 8, 3), numpy.uint8),
            ValueError, r"height, width, 4")


def test_non_3d_buffer_is_rejected(make_backend):
    backend, _user32, gdi32 = make_backend()
    _reject(backend, gdi32, numpy.zeros((8, 8), numpy.uint8),
            ValueError, r"height, width, 4")


def test_wrong_dtype_is_rejected(make_backend):
    backend, _user32, gdi32 = make_backend()
    _reject(backend, gdi32, numpy.zeros((8, 8, 4), numpy.float64),
            ValueError, "dtype uint8")


def test_non_array_is_rejected(make_backend):
    backend, _user32, gdi32 = make_backend()
    _reject(backend, gdi32, [[0, 0, 0, 0]], TypeError, "ndarray")


def test_empty_buffer_is_rejected(make_backend):
    backend, _user32, gdi32 = make_backend()
    _reject(backend, gdi32, numpy.zeros((0, 8, 4), numpy.uint8),
            ValueError, "positive")


def test_a_valid_destination_still_captures(make_backend):
    # The guard must not have made the ordinary path stricter.
    backend, user32, gdi32 = make_backend()
    img = _capture(backend, 0, 0, 12, 10)
    assert img.shape == (10, 12, 4)
    assert len(gdi32.blits) == 1
    assert numpy.array_equal(img, user32.screen[0:10, 0:12])
