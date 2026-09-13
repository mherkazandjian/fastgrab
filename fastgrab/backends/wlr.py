"""Wayland backend using ``wlr-screencopy-unstable-v1``.

Works on wlroots-based compositors (Sway, Hyprland, river, niri, cage, …)
that advertise the ``zwlr_screencopy_manager_v1`` global. Does NOT work on
GNOME (Mutter) or KDE (KWin) — for those, the portal backend is the
intended fallback.

V1 captures from a single output. Selection order:
``$FASTGRAB_OUTPUT`` (if set, matches output ``name``) → first output
advertised. Multi-monitor composition is out of scope for this version.

``capture_output_region`` takes its region in **output logical
coordinates**, while a fastgrab bbox is device pixels and
:meth:`WlrBackend.resolution` reports the ``wl_output`` mode size. Those
agree only on an unscaled output. The gap is closed by tracking the
output's logical geometry through ``zxdg_output_manager_v1`` and
converting at this boundary — see :func:`_plan_region` for the
conversion and exactly what it is allowed to assume.
"""
import mmap
import os
import weakref
import struct  # noqa: F401  (reserved for future DMA-BUF / extended formats)

import numpy as np

from pywayland.client import Display
from pywayland.protocol.wayland import WlOutput, WlShm

try:
    from pywayland.protocol.xdg_output_unstable_v1 import ZxdgOutputManagerV1
except ImportError:  # pragma: no cover — older than the pywayland the extra pins
    # Without xdg_output the output's logical size is unknowable, and
    # sub-region capture degrades to a full-output capture plus a crop.
    # See WlrBackend._subregion_plan.
    ZxdgOutputManagerV1 = None

from .base import BaseBackend
from .protocols.wlr_screencopy_unstable_v1 import ZwlrScreencopyManagerV1


# wl_shm format constants — see wayland.xml
_WL_SHM_FORMAT_ARGB8888 = 0
_WL_SHM_FORMAT_XRGB8888 = 1
# Both of the above are little-endian BGRA in memory on x86_64,
# matching the byte order our X11 backend already produces.

_FRAME_VERSION = 3

# zxdg_output_manager_v1 is at version 3; logical_position and
# logical_size have been there since version 1, so binding low is safe.
_XDG_OUTPUT_VERSION = 3


class _OutputState:
    """Mutable accumulator for a single ``wl_output``'s geometry events."""
    __slots__ = ("proxy", "name", "mode_w", "mode_h", "scale", "transform",
                 "logical_x", "logical_y", "logical_w", "logical_h",
                 "xdg", "done")

    def __init__(self, proxy):
        self.proxy = proxy
        self.name = None
        self.mode_w = 0
        self.mode_h = 0
        self.scale = 1
        # WL_OUTPUT_TRANSFORM_NORMAL; anything else rotates or flips the
        # logical coordinate space relative to the backing store.
        self.transform = 0
        # xdg_output's logical geometry, the unit capture_output_region
        # speaks. 0 means "never told": a compositor need not advertise
        # zxdg_output_manager_v1 at all.
        self.logical_x = 0
        self.logical_y = 0
        self.logical_w = 0
        self.logical_h = 0
        self.xdg = None
        self.done = False


class _FrameState:
    """Per-capture event accumulator."""
    __slots__ = ("fmt", "w", "h", "stride", "buffer_done",
                 "flags", "ready", "failed", "offered")

    def __init__(self):
        self.fmt = None
        self.w = 0
        self.h = 0
        self.stride = 0
        self.buffer_done = False
        self.flags = 0
        self.ready = False
        self.failed = False
        # Formats the compositor offered that are not BGRA in memory,
        # kept so a refusal can name them.
        self.offered = []


class _RegionPlan:
    """How to ask for a device-pixel bbox, and what to do with the answer.

    ``region`` is the ``(x, y, width, height)`` to hand to
    ``capture_output_region`` in output *logical* coordinates, or
    ``None`` to capture the whole output instead. ``crop_x``/``crop_y``
    say where the requested bbox starts inside the frame that comes
    back, in device pixels. ``exp_w``/``exp_h`` is the device-pixel frame
    size the mapping predicts; it is checked against what the compositor
    actually returned before a byte is copied, which is what makes the
    rest of the plan trustworthy rather than merely plausible.
    """
    __slots__ = ("region", "crop_x", "crop_y", "exp_w", "exp_h")

    def __init__(self, region, crop_x, crop_y, exp_w, exp_h):
        self.region = region
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.exp_w = exp_w
        self.exp_h = exp_h


def _ceil_div(numerator, denominator):
    """Integer ceiling division, for non-negative operands."""
    return -(-numerator // denominator)


def _uniform_integer_scale(mode_w, mode_h, logical_w, logical_h):
    """Return ``k`` when ``device == logical * k`` exactly on both axes.

    ``None`` when the logical geometry is unknown, or when the ratio is
    not one and the same whole number on both axes — a fractionally
    scaled output, or one whose logical size the compositor rounded.
    """
    if logical_w <= 0 or logical_h <= 0:
        return None
    if mode_w % logical_w or mode_h % logical_h:
        return None
    scale = mode_w // logical_w
    if scale < 1 or mode_h // logical_h != scale:
        return None
    return scale


def _plan_region(x, y, w, h, mode_w, mode_h, logical_w, logical_h):
    """Map a device-pixel bbox onto a logical region plus a crop.

    Two cases, and the split is about what can be *proved*, not about
    what is likely.

    **Exact integer scale.** wlroots turns a region request into a
    device-pixel box by multiplying each component by the output scale
    and truncating (``buffer_box.x *= output->scale``, an ``int``
    left-hand side and a ``float`` right-hand side). Truncation is a
    no-op for a whole-number scale, so the frame that comes back starts
    at exactly ``logical_origin * k``: floor the requested origin into
    logical units, ceil the far edge so every requested pixel stays
    covered, and crop the surplus off the returned frame.

    That ``k`` is derived, not observed. ``wl_output.scale`` is a
    ``ceil()``'d integer, and xdg_output's logical size is itself
    ``trunc(mode / scale)``, which only bounds the true scale ``s`` from
    above (``s <= mode / logical == k``). The frame-size check in
    :meth:`WlrBackend.screenshot` closes the gap: the compositor returns
    ``trunc(lw * s)`` where this plan predicts ``lw * k``, and those
    agree only when ``s >= k``. A matching size therefore *proves*
    ``s == k``, and with it the origin. An output at, say, 1.999 on a
    mode whose logical size happens to divide evenly returns a short
    frame and is rejected rather than silently mis-cropped.

    **Anything else** — a fractional scale, or axes that disagree. There
    the device origin of a region frame is ``trunc(lx * s)`` for a float
    ``s`` that cannot be recovered exactly from the integers the
    protocol hands us, so any origin but zero risks being off by a
    pixel. Rather than guess, capture the whole output — which is
    ``mode_w x mode_h`` device pixels by definition, with no scale
    arithmetic involved — and crop the bbox out of it. Correct, at the
    cost of copying the whole output for a small bbox: the same trade
    this backend used to ask callers to make by hand.
    """
    scale = _uniform_integer_scale(mode_w, mode_h, logical_w, logical_h)
    if scale is None:
        return _RegionPlan(None, x, y, mode_w, mode_h)

    lx = x // scale
    ly = y // scale
    lw = _ceil_div(x + w, scale) - lx
    lh = _ceil_div(y + h, scale) - ly
    return _RegionPlan(
        (lx, ly, lw, lh),
        crop_x=x - lx * scale,
        crop_y=y - ly * scale,
        exp_w=lw * scale,
        exp_h=lh * scale,
    )


def _release_shm(mm, fd, wl_buffer):
    """Free one SHM frame buffer: the compositor's claim first, then ours.

    The order is the whole point. A ``wl_buffer`` the compositor still
    owns keeps the compositor's own mapping of the memfd alive, so
    closing our file descriptor first releases no memory at all —
    measured against cage, 50 dropped buffers held 175.5 MiB of system
    Shmem, and closing every descriptor by hand freed none of it. Sending
    ``destroy`` brought the same 50 down to 0.8 MiB.

    Nothing here may raise. This runs from a :mod:`weakref` finalizer,
    where an exception is printed and swallowed at an arbitrary point in
    someone else's call stack. A dead connection is not a leak either —
    the compositor drops every resource it held for us when the socket
    closes.
    """
    try:
        wl_buffer.destroy()
    except Exception:
        pass
    try:
        mm.close()
    except Exception:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


class _ShmBuffer:
    """Owns one memfd + mmap + wl_buffer, and frees them when dropped.

    The finalizer lives here rather than on :class:`WlrBackend` so that
    dropping a backend releases its frame buffer without the backend
    needing a ``__del__``. That separation is the point: the backend is
    entangled with pywayland's connection-wide state, whose finalisation
    order at interpreter shutdown is exactly what makes ``__del__``
    unsafe here, while these three handles are this object's own and
    nobody else's.

    ``atexit`` is turned off for the same reason. At shutdown the kernel
    reclaims the descriptor and the mapping regardless, and the
    compositor frees everything when the socket closes, so firing then
    buys nothing and would push a request through a display that may
    already be half torn down. The leak worth fixing is the one that
    accumulates *while* the process runs.
    """

    # __weakref__ is not optional here: weakref.finalize takes a weak
    # reference to this object, and a __slots__ class does not get one
    # unless it is asked for. Without it construction raises outright.
    __slots__ = ("mm", "fd", "wl", "w", "h", "stride", "fmt", "_finalize",
                 "__weakref__")

    def __init__(self, mm, fd, wl, w, h, stride, fmt):
        self.mm = mm
        self.fd = fd
        self.wl = wl
        self.w = w
        self.h = h
        self.stride = stride
        self.fmt = fmt
        # The finalizer is handed the raw handles, never ``self``: a
        # finalizer that referenced its own object would keep that object
        # reachable and so could never run.
        self._finalize = weakref.finalize(self, _release_shm, mm, fd, wl)
        self._finalize.atexit = False

    def matches(self, w, h, stride, fmt):
        return (self.w, self.h, self.stride, self.fmt) == (w, h, stride, fmt)

    def close(self):
        """Free now. Idempotent — a finalizer only ever fires once."""
        self._finalize()


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
        # Set instead of binding when the advertised screencopy version is
        # too old for the handshake below, so the refusal can name it.
        screencopy_version = [None]
        xdg_output_manager = [None]

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
                proxy.dispatcher["geometry"] = lambda p, gx, gy, pw, ph, \
                    subpixel, make, model, transform, s=state: (
                    WlrBackend._on_output_geometry(s, transform)
                )
                proxy.dispatcher["done"] = lambda p, s=state: (
                    WlrBackend._on_output_done(s)
                )
                outputs.append(state)
            elif interface == "wl_shm":
                shm[0] = registry.bind(name, WlShm, min(version, 1))
            elif interface == "zwlr_screencopy_manager_v1":
                # The capture loop waits for `buffer_done`, which the
                # protocol only introduced in version 3. Bound below
                # that, the compositor sends `buffer` and then waits for
                # `copy` while we wait for an event it will never send --
                # a capture that blocks forever rather than failing.
                # wlroots has shipped version 3 since 2020, so refusing
                # is proportionate; supporting the legacy handshake would
                # mean an untestable code path for compositors that
                # effectively no longer exist.
                if version < _FRAME_VERSION:
                    screencopy_version[0] = version
                    return
                screencopy[0] = registry.bind(
                    name, ZwlrScreencopyManagerV1, min(version, _FRAME_VERSION)
                )
            elif (interface == "zxdg_output_manager_v1"
                    and ZxdgOutputManagerV1 is not None):
                xdg_output_manager[0] = registry.bind(
                    name, ZxdgOutputManagerV1,
                    min(version, _XDG_OUTPUT_VERSION),
                )

        registry = display.get_registry()
        registry.dispatcher["global"] = _on_global
        display.roundtrip()

        # xdg_output has to be a second pass: get_xdg_output needs the
        # manager *and* the wl_output, and the registry is free to
        # announce them in either order.
        if xdg_output_manager[0] is not None:
            for state in outputs:
                WlrBackend._bind_xdg_output(xdg_output_manager[0], state)

        display.roundtrip()
        display.roundtrip()  # let mode/name/done + logical_size settle

        if screencopy[0] is None and screencopy_version[0] is not None:
            raise RuntimeError(
                "compositor advertises zwlr_screencopy_manager_v1 version "
                "{}, but fastgrab needs version {}: the capture handshake "
                "waits for the `buffer_done` event, which the protocol only "
                "added in version 3. Binding lower would block forever "
                "waiting for an event the compositor never sends."
                .format(screencopy_version[0], _FRAME_VERSION)
            )
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
    def _bind_xdg_output(manager, state):
        """Create an ``xdg_output`` for ``state`` and wire up its events."""
        xdg = manager.get_xdg_output(state.proxy)
        xdg.dispatcher["logical_position"] = lambda o, lx, ly, s=state: (
            WlrBackend._on_xdg_logical_position(s, lx, ly)
        )
        xdg.dispatcher["logical_size"] = lambda o, lw, lh, s=state: (
            WlrBackend._on_xdg_logical_size(s, lw, lh)
        )
        state.xdg = xdg
        return xdg

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
    def _on_output_geometry(state, transform):
        state.transform = transform

    @staticmethod
    def _on_output_done(state):
        state.done = True

    @staticmethod
    def _on_xdg_logical_position(state, x, y):
        state.logical_x = x
        state.logical_y = y

    @staticmethod
    def _on_xdg_logical_size(state, width, height):
        state.logical_w = width
        state.logical_h = height

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

    def refresh(self):
        """Re-read output geometry and re-select the output.

        Unlike x11 and windows, this backend does not query the
        compositor per call: ``resolution()`` returns ``wl_output`` mode
        fields latched from events at connect time, so a mode change is
        invisible until the display is dispatched again. Two round trips
        let pending ``mode``/``name``/``done`` events settle, matching
        what ``_connect_singleton`` does. The ``xdg_output`` objects are
        long-lived, so a scale change re-sends ``logical_size`` over the
        same round trips.

        Mode changes, scale changes and a different ``FASTGRAB_OUTPUT``
        choice among the already-bound outputs are picked up. An output
        that has since been unplugged is *not*: that needs registry
        ``global_remove`` tracking, which this backend does not do.
        """
        self._display.roundtrip()
        self._display.roundtrip()
        self._output = self._select_output()

    def resolution(self):
        return (self._output.mode_w, self._output.mode_h)

    def bytes_per_pixel(self):
        return 4

    def _subregion_plan(self, x, y, w, h):
        """Plan a sub-region capture of the device-pixel bbox ``x, y, w, h``.

        Raises :class:`NotImplementedError` for the one mapping this
        backend still refuses outright: a rotated or flipped output.
        wlroots applies the output transform *before* the scale, so a
        region on a transformed output lands on a transposed backing
        store — and nothing here models that, so guessing would be worse
        than refusing.
        """
        out = self._output
        if out.transform != 0:
            raise NotImplementedError(
                "sub-region capture is not supported on output {!r} "
                "(transform {}): wlr-screencopy takes the region in "
                "logical coordinates and wlroots applies the output "
                "transform before the scale, so on a rotated or flipped "
                "output the region lands on a transposed backing store — "
                "a 20x10 request comes back 10x20. fastgrab does not "
                "model output transforms. Capture the full output and "
                "slice the returned array instead."
                .format(out.name, out.transform)
            )

        logical_w = out.logical_w or 0
        logical_h = out.logical_h or 0
        if logical_w > 0 and logical_h > 0:
            return _plan_region(x, y, w, h, out.mode_w, out.mode_h,
                                logical_w, logical_h)

        # No xdg_output: the compositor never told us the logical size.
        if out.scale != 1:
            # wl_output.scale is ceil() of the real scale, an upper bound
            # and nothing more — not enough to place a region. Take the
            # whole output, whose size needs no scale arithmetic, and crop.
            return _RegionPlan(None, x, y, out.mode_w, out.mode_h)

        # Scale 1 and no logical size: assume the identity mapping and
        # let the frame-size check adjudicate. It has to, because ceil()
        # collapses every scale in (0, 1] onto the same reported 1.
        return _RegionPlan((x, y, w, h), 0, 0, w, h)

    def screenshot(self, x, y, img):
        self._check_open()
        h, w, _ = img.shape
        full_w, full_h = self.resolution()
        if x == 0 and y == 0 and w == full_w and h == full_h:
            # The whole output: no region, so no logical conversion at all.
            plan = _RegionPlan(None, 0, 0, w, h)
        else:
            plan = self._subregion_plan(x, y, w, h)

        if plan.region is None:
            frame = self._screencopy.capture_output(0, self._output.proxy)
        else:
            frame = self._screencopy.capture_output_region(
                0, self._output.proxy, *plan.region
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
        if state.fmt is None:
            # Every offered format had a memory layout that is not BGRA.
            # Refused by name rather than copied out as if it were: an
            # ABGR8888 frame read as BGRA returns red pixels as blue.
            offered = ", ".join(
                "0x{:08x}".format(f) for f in state.offered
            ) or "none"
            frame.destroy()
            raise RuntimeError(
                "compositor offered no BGRA-compatible buffer format for "
                "this frame (offered: {}). fastgrab needs "
                "WL_SHM_FORMAT_ARGB8888 (0) or WL_SHM_FORMAT_XRGB8888 (1), "
                "whose bytes are already B, G, R, A.".format(offered)
            )

        # Check the size the compositor actually returned rather than
        # trusting the scale it announced. wl_output.scale is an integer
        # event reported as ceil() of the real scale, so an output at
        # 0.75 announces 1 and looks like identity; xdg_output's logical
        # size only bounds the real scale from above. This check is what
        # turns _plan_region's derived mapping into a proved one — and
        # without it `img[:] = arr` would *broadcast* a too-small frame
        # over the destination instead of failing, filling a 2x2 request
        # from a single pixel.
        if state.w != plan.exp_w or state.h != plan.exp_h:
            frame.destroy()
            raise NotImplementedError(
                "compositor returned a {}x{} frame where the region "
                "mapping predicted {}x{} device pixels, for a {}x{} "
                "capture at ({}, {}) on output {!r} (mode {}x{}, logical "
                "{}x{}, wl_output.scale {}, transform {}). The region is "
                "interpreted in logical coordinates, and wl_output.scale "
                "is an integer reported as ceil() of the real scale, so a "
                "fractionally scaled output cannot always be detected up "
                "front. Capture the full output and slice the returned "
                "array instead."
                .format(state.w, state.h, plan.exp_w, plan.exp_h, w, h,
                        x, y, self._output.name,
                        self._output.mode_w, self._output.mode_h,
                        self._output.logical_w, self._output.logical_h,
                        self._output.scale, self._output.transform)
            )

        wl_buffer, mm = self._ensure_buffer(state.fmt, state.w, state.h, state.stride)

        frame.copy(wl_buffer)
        while not state.ready and not state.failed:
            self._display.dispatch(block=True)
        if state.failed:
            frame.destroy()
            raise RuntimeError("wlr-screencopy frame copy failed")

        # Memcpy from mmap into the caller's ndarray, accounting for
        # stride. The width and stride from the buffer event drive this,
        # never the requested size: the frame is legitimately bigger than
        # the bbox whenever the logical region had to be rounded outward,
        # or the whole output was taken to crop out of.
        frame_bytes = bytes(mm[: state.stride * state.h])
        arr = np.frombuffer(frame_bytes, dtype="uint8").reshape(
            state.h, state.stride
        )[:, : state.w * 4].reshape(state.h, state.w, 4)
        if state.flags & 0x1:  # Y_INVERT
            # Flip before cropping — the flag describes the frame, so the
            # crop offsets only count from the top left once it is upright.
            arr = np.flipud(arr)
        # Caller's ndarray was sized (h, w, 4) by the wrapper; copy the
        # requested bbox out of the frame. The slice is exact, so a
        # mismatch raises instead of broadcasting.
        img[:] = arr[plan.crop_y:plan.crop_y + h, plan.crop_x:plan.crop_x + w]

        frame.destroy()

    # -------- frame event handlers --------

    @staticmethod
    def _on_frame_buffer(state, fmt, w, h, stride):
        """Accept only a format whose memory layout is already BGRA.

        A version-3 compositor sends one of these per supported buffer
        type, so this runs several times and must pick, not merely
        record. ARGB8888 and XRGB8888 are 32-bit little-endian words, so
        their bytes land B, G, R, A — exactly the public contract.

        The previous condition began ``state.fmt is None or ...``, which
        accepted whatever arrived *first* whether or not it was one of
        those. A compositor offering ABGR8888 (bytes R, G, B, A) had its
        frames copied out as if they were BGRA, so red came back as blue.
        Every other format is refused by name in ``screenshot()`` rather
        than silently mis-read.
        """
        if fmt not in (_WL_SHM_FORMAT_ARGB8888, _WL_SHM_FORMAT_XRGB8888):
            state.offered.append(fmt)
            return
        # Both are BGRA in memory; keep the first acceptable one.
        if state.fmt is None:
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
            if self._buf.matches(w, h, stride, fmt):
                return self._buf.wl, self._buf.mm
            self._buf.close()
            self._buf = None

        size = stride * h
        fd = os.memfd_create("fastgrab-wlr", 0)
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                       flags=mmap.MAP_SHARED)
        pool = self._shm.create_pool(fd, size)
        wl_buffer = pool.create_buffer(0, w, h, stride, fmt)
        pool.destroy()

        self._buf = _ShmBuffer(mm, fd, wl_buffer, w, h, stride, fmt)
        return wl_buffer, mm

    def close(self):
        """Release this instance's SHM frame buffer.

        Optional — dropping the backend frees the same buffer through
        :class:`_ShmBuffer`'s finalizer. Call it to pick the moment.

        Still no ``__del__`` on the backend itself: pywayland's C-level
        state has its own lifecycle and finalising it in arbitrary GC
        order can segfault. Only the buffer is ours to free; the display
        is process-wide and outlives every instance.
        """
        if self._buf is not None:
            self._buf.close()
            self._buf = None
        self._closed = True
