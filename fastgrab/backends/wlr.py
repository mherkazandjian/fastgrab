"""Wayland backend using ``wlr-screencopy-unstable-v1``.

Works on wlroots-based compositors (Sway, Hyprland, river, niri, cage, …)
that advertise the ``zwlr_screencopy_manager_v1`` global. Does NOT work on
GNOME (Mutter) or KDE (KWin) — for those, the portal backend is the
intended fallback.

V1 captures from a single output. Selection order:
``$FASTGRAB_OUTPUT`` (if set, matches output ``name``) → first output
advertised. Multi-monitor composition is out of scope for this version.
"""
import mmap
import os
import struct  # noqa: F401  (reserved for future DMA-BUF / extended formats)

import numpy as np

from pywayland.client import Display
from pywayland.protocol.wayland import WlOutput, WlShm

from .base import BaseBackend
from .protocols.wlr_screencopy_unstable_v1 import ZwlrScreencopyManagerV1


# wl_shm format constants — see wayland.xml
_WL_SHM_FORMAT_ARGB8888 = 0
_WL_SHM_FORMAT_XRGB8888 = 1
# Both of the above are little-endian BGRA in memory on x86_64,
# matching the byte order our X11 backend already produces.

_FRAME_VERSION = 3


class _OutputState:
    """Mutable accumulator for a single ``wl_output``'s geometry events."""
    __slots__ = ("proxy", "name", "mode_w", "mode_h", "scale", "done")

    def __init__(self, proxy):
        self.proxy = proxy
        self.name = None
        self.mode_w = 0
        self.mode_h = 0
        self.scale = 1
        self.done = False


class _FrameState:
    """Per-capture event accumulator."""
    __slots__ = ("fmt", "w", "h", "stride", "buffer_done",
                 "flags", "ready", "failed")

    def __init__(self):
        self.fmt = None
        self.w = 0
        self.h = 0
        self.stride = 0
        self.buffer_done = False
        self.flags = 0
        self.ready = False
        self.failed = False


# The wayland connection + globals are shared across all WlrBackend
# instances in this process. pywayland's C-level state lifecycle does not
# survive multiple parallel Display.connect / disconnect cycles cleanly
# (we segfault during GC when an earlier connection's proxies finalise
# while a newer connection is mid-event). One connection per process is
# also a faithful model — there is at most one Wayland session.
_SINGLETON_STATE = None  # ("ok", display, shm, screencopy, outputs) | ("err", exc)


class WlrBackend(BaseBackend):
    def __init__(self):
        global _SINGLETON_STATE
        if _SINGLETON_STATE is None:
            try:
                _SINGLETON_STATE = ("ok",) + self._connect_singleton()
            except Exception as exc:  # cache failure too; same outcome on retry
                _SINGLETON_STATE = ("err", exc)

        if _SINGLETON_STATE[0] == "err":
            raise _SINGLETON_STATE[1]

        _, self._display, self._shm, self._screencopy, self._outputs = (
            _SINGLETON_STATE
        )
        self._buf = None  # per-instance SHM buffer cache
        self._output = self._select_output()

    @staticmethod
    def _connect_singleton():
        display = Display()
        display.connect()

        outputs = []
        shm = [None]
        screencopy = [None]

        def _on_global(registry, name, interface, version):
            if interface == "wl_output":
                proxy = registry.bind(name, WlOutput, min(version, 4))
                state = _OutputState(proxy)
                proxy.dispatcher["mode"] = lambda p, flags, w, h, refresh, s=state: (
                    WlrBackend._on_output_mode(s, flags, w, h)
                )
                proxy.dispatcher["name"] = lambda p, n, s=state: (
                    WlrBackend._on_output_name(s, n)
                )
                proxy.dispatcher["scale"] = lambda p, factor, s=state: (
                    WlrBackend._on_output_scale(s, factor)
                )
                proxy.dispatcher["done"] = lambda p, s=state: (
                    WlrBackend._on_output_done(s)
                )
                outputs.append(state)
            elif interface == "wl_shm":
                shm[0] = registry.bind(name, WlShm, min(version, 1))
            elif interface == "zwlr_screencopy_manager_v1":
                screencopy[0] = registry.bind(
                    name, ZwlrScreencopyManagerV1, min(version, _FRAME_VERSION)
                )

        registry = display.get_registry()
        registry.dispatcher["global"] = _on_global
        display.roundtrip()
        display.roundtrip()  # let wl_output mode/name/done events settle

        if screencopy[0] is None:
            raise RuntimeError(
                "compositor does not advertise zwlr_screencopy_manager_v1; "
                "fall back to the portal backend on GNOME/KDE"
            )
        if shm[0] is None:
            raise RuntimeError("compositor does not advertise wl_shm")
        if not outputs:
            raise RuntimeError("compositor advertised no wl_output")

        return display, shm[0], screencopy[0], outputs

    # -------- output state event handlers --------

    @staticmethod
    def _on_output_mode(state, flags, width, height):
        # current mode is flagged with WL_OUTPUT_MODE_CURRENT (0x1)
        if flags & 0x1:
            state.mode_w = width
            state.mode_h = height

    @staticmethod
    def _on_output_name(state, name):
        state.name = name

    @staticmethod
    def _on_output_scale(state, factor):
        state.scale = factor

    @staticmethod
    def _on_output_done(state):
        state.done = True

    def _select_output(self):
        wanted = os.environ.get("FASTGRAB_OUTPUT")
        if wanted:
            for o in self._outputs:
                if o.name == wanted:
                    return o
            raise RuntimeError(
                "FASTGRAB_OUTPUT={!r} not found; available: {}".format(
                    wanted, [o.name for o in self._outputs]
                )
            )
        return self._outputs[0]

    # -------- BaseBackend API --------

    def resolution(self):
        return (self._output.mode_w, self._output.mode_h)

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        h, w, _ = img.shape
        full_w, full_h = self.resolution()
        if x == 0 and y == 0 and w == full_w and h == full_h:
            frame = self._screencopy.capture_output(0, self._output.proxy)
        else:
            frame = self._screencopy.capture_output_region(
                0, self._output.proxy, x, y, w, h
            )

        state = _FrameState()
        frame.dispatcher["buffer"] = lambda f, fmt, fw, fh, st: (
            self._on_frame_buffer(state, fmt, fw, fh, st)
        )
        frame.dispatcher["linux_dmabuf"] = lambda f, fmt, fw, fh: None
        frame.dispatcher["buffer_done"] = lambda f: (
            self._on_frame_buffer_done(state)
        )
        frame.dispatcher["flags"] = lambda f, flags: (
            self._on_frame_flags(state, flags)
        )
        frame.dispatcher["ready"] = lambda f, hi, lo, ns: (
            self._on_frame_ready(state)
        )
        frame.dispatcher["failed"] = lambda f: (
            self._on_frame_failed(state)
        )

        # Pull buffer + buffer_done events.
        while not state.buffer_done and not state.failed:
            self._display.dispatch(block=True)
        if state.failed:
            frame.destroy()
            raise RuntimeError("wlr-screencopy frame failed before buffer info")

        wl_buffer, mm = self._ensure_buffer(state.fmt, state.w, state.h, state.stride)

        frame.copy(wl_buffer)
        while not state.ready and not state.failed:
            self._display.dispatch(block=True)
        if state.failed:
            frame.destroy()
            raise RuntimeError("wlr-screencopy frame copy failed")

        # Memcpy from mmap into the caller's ndarray, accounting for stride.
        frame_bytes = bytes(mm[: state.stride * state.h])
        arr = np.frombuffer(frame_bytes, dtype="uint8").reshape(
            state.h, state.stride
        )[:, : state.w * 4].reshape(state.h, state.w, 4)
        if state.flags & 0x1:  # Y_INVERT
            arr = np.flipud(arr)
        # Caller's ndarray was sized (h, w, 4) by the wrapper; copy in.
        img[:] = arr

        frame.destroy()

    # -------- frame event handlers --------

    @staticmethod
    def _on_frame_buffer(state, fmt, w, h, stride):
        # Prefer the SHM buffer event over linux_dmabuf — we always use SHM.
        if state.fmt is None or fmt in (
            _WL_SHM_FORMAT_ARGB8888,
            _WL_SHM_FORMAT_XRGB8888,
        ):
            state.fmt = fmt
            state.w = w
            state.h = h
            state.stride = stride

    @staticmethod
    def _on_frame_buffer_done(state):
        state.buffer_done = True

    @staticmethod
    def _on_frame_flags(state, flags):
        state.flags = flags

    @staticmethod
    def _on_frame_ready(state):
        state.ready = True

    @staticmethod
    def _on_frame_failed(state):
        state.failed = True

    # -------- SHM buffer cache --------

    def _ensure_buffer(self, fmt, w, h, stride):
        """Reuse the SHM-backed wl_buffer if dimensions match the last call."""
        if self._buf is not None:
            old_mm, old_fd, old_wl, old_w, old_h, old_stride, old_fmt = self._buf
            if (old_w, old_h, old_stride, old_fmt) == (w, h, stride, fmt):
                return old_wl, old_mm
            # dispose of old
            old_wl.destroy()
            old_mm.close()
            os.close(old_fd)
            self._buf = None

        size = stride * h
        fd = os.memfd_create("fastgrab-wlr", 0)
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                       flags=mmap.MAP_SHARED)
        pool = self._shm.create_pool(fd, size)
        wl_buffer = pool.create_buffer(0, w, h, stride, fmt)
        pool.destroy()

        self._buf = (mm, fd, wl_buffer, w, h, stride, fmt)
        return wl_buffer, mm

    # Intentionally no __del__ — pywayland's C-level state has its own
    # lifecycle and finalising in arbitrary GC order can segfault. The
    # OS reclaims the fd / mmap when the process exits.
