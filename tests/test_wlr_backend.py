"""Fake-state tests for the wlr backend that need no compositor.

``docker compose run --rm test-wayland`` exercises the real thing under
headless ``cage``, but that session has a single untransformed output
and never changes mode, so the paths :meth:`WlrBackend.refresh` and the
transform refusal exist for are unreachable there. The scale-2 mapping
*is* reachable there (see ``tests/test_integration_wlr.py``), but only
for the one scale ``wlr-randr`` is asked to set. These drive the same
code over hand-built output state instead.
"""
import gc
import mmap
import os
from types import SimpleNamespace

import numpy
import pytest

pytest.importorskip(
    "pywayland", reason="the wlr backend is behind the [wayland] extra"
)

from fastgrab.backends.wlr import (  # noqa: E402
    WlrBackend, _ShmBuffer, _plan_region, _release_shm, _uniform_integer_scale,
)

# Hand-built output state only — no compositor, no display server.
pytestmark = pytest.mark.no_display


def _output(name="HDMI-1", mode_w=100, mode_h=50, scale=1, transform=0,
            logical_w=0, logical_h=0):
    """An output as ``_connect_singleton`` would have accumulated it.

    ``logical_w``/``logical_h`` are the ``xdg_output.logical_size`` the
    compositor reported; 0 means it never did — a compositor need not
    advertise ``zxdg_output_manager_v1``.
    """
    return SimpleNamespace(proxy=object(), name=name, mode_w=mode_w,
                           mode_h=mode_h, scale=scale, transform=transform,
                           logical_x=0, logical_y=0,
                           logical_w=logical_w, logical_h=logical_h,
                           xdg=None)


def _scaled_output(mode_w, mode_h, logical_w, logical_h, **kw):
    """An output whose logical size is known, as xdg_output reports it."""
    return _output(mode_w=mode_w, mode_h=mode_h,
                   logical_w=logical_w, logical_h=logical_h, **kw)


def _fake_backend(outputs, on_roundtrip=None):
    """A WlrBackend over fake output state, bypassing the connection."""
    backend = object.__new__(WlrBackend)
    roundtrips = []

    def roundtrip():
        roundtrips.append(1)
        if on_roundtrip is not None:
            on_roundtrip()

    backend._display = SimpleNamespace(roundtrip=roundtrip)
    backend._outputs = outputs
    backend._output = outputs[0]
    return backend, roundtrips


def test_refresh_picks_up_a_mode_change():
    """resolution() reads fields latched from events, not live state."""
    output = _output()

    def new_mode_arrives():
        output.mode_w, output.mode_h = 200, 120

    backend, roundtrips = _fake_backend([output], new_mode_arrives)
    assert backend.resolution() == (100, 50)

    backend.refresh()
    assert backend.resolution() == (200, 120)
    # two, matching what _connect_singleton does to let events settle
    assert len(roundtrips) == 2


def test_refresh_reselects_the_requested_output(monkeypatch):
    first = _output()
    second = _output(name="DP-1", mode_w=300, mode_h=150)
    backend, _ = _fake_backend([first, second])
    assert backend._output is first

    monkeypatch.setenv("FASTGRAB_OUTPUT", "DP-1")
    backend.refresh()
    assert backend._output is second
    assert backend.resolution() == (300, 150)


def test_refresh_raises_for_an_unknown_requested_output(monkeypatch):
    backend, _ = _fake_backend([_output()])

    monkeypatch.setenv("FASTGRAB_OUTPUT", "DP-9")
    with pytest.raises(RuntimeError, match="not found"):
        backend.refresh()


# -------- the device-pixel -> logical conversion, on its own --------


def test_uniform_integer_scale_recognises_a_whole_number_ratio():
    assert _uniform_integer_scale(200, 100, 100, 50) == 2
    assert _uniform_integer_scale(100, 50, 100, 50) == 1
    assert _uniform_integer_scale(300, 150, 100, 50) == 3


@pytest.mark.parametrize("mode, logical, why", [
    ((1920, 1080), (1280, 720), "1.5 is not a whole number"),
    ((1920, 1080), (1097, 617), "1.75, and the division was truncated"),
    ((1920, 1080), (960, 1080), "the two axes disagree"),
    ((1920, 1080), (0, 0), "the compositor never sent a logical size"),
])
def test_uniform_integer_scale_refuses_anything_else(mode, logical, why):
    assert _uniform_integer_scale(*mode, *logical) is None, why


def test_plan_region_at_scale_1_is_the_identity():
    """An unscaled output is the case the old pass-through got right."""
    plan = _plan_region(20, 10, 20, 10, 100, 50, 100, 50)
    assert plan.region == (20, 10, 20, 10)
    assert (plan.crop_x, plan.crop_y) == (0, 0)
    assert (plan.exp_w, plan.exp_h) == (20, 10)


def test_plan_region_at_scale_2_halves_the_request():
    """The issue's case: a 20x10 device request is 10x5 logical.

    Sending 20x10 through unchanged is what issue #38 is about — the
    compositor reads it as logical, doubles it, and returns a 40x20
    frame that does not fit the 20x10 destination.
    """
    plan = _plan_region(20, 10, 20, 10, 200, 100, 100, 50)
    assert plan.region == (10, 5, 10, 5)
    assert (plan.exp_w, plan.exp_h) == (20, 10)


def test_plan_region_at_scale_2_displaces_the_origin_too():
    """Not just the size: the origin is in logical units as well.

    A bbox at device (60, 30) starts at logical (30, 15). Passing 60,30
    straight through would capture from twice as far into the output.
    """
    plan = _plan_region(60, 30, 20, 10, 200, 100, 100, 50)
    assert plan.region[:2] == (30, 15)
    assert (plan.crop_x, plan.crop_y) == (0, 0)


def test_plan_region_rounds_the_logical_region_outward():
    """A bbox not aligned to the scale must still be fully covered.

    Device x=21 sits inside logical column 10 (which covers device
    21..22 at scale 2), and the far edge 41 needs logical 21. So the
    region is 10..20 wide, i.e. 11 logical columns, and the surplus
    device pixel on the left is cropped off afterwards.
    """
    plan = _plan_region(21, 11, 20, 10, 200, 100, 100, 50)
    assert plan.region == (10, 5, 11, 6)
    # floor(21/2) * 2 == 20, so the bbox starts one device pixel in
    assert (plan.crop_x, plan.crop_y) == (1, 1)
    assert (plan.exp_w, plan.exp_h) == (22, 12)
    # the rounded-outward frame really does contain the whole bbox
    assert plan.crop_x + 20 <= plan.exp_w
    assert plan.crop_y + 10 <= plan.exp_h


@pytest.mark.parametrize("x, y, w, h", [
    (0, 0, 1, 1), (1, 1, 1, 1), (3, 7, 5, 9), (37, 41, 63, 59),
    (0, 0, 200, 100), (199, 99, 1, 1),
])
@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_plan_region_always_covers_the_requested_bbox(x, y, w, h, scale):
    """Whatever the alignment, the crop must land inside the frame."""
    mode_w, mode_h = 200 * scale, 100 * scale
    plan = _plan_region(x * scale, y * scale, w, h, mode_w, mode_h, 200, 100)
    assert plan.crop_x >= 0 and plan.crop_y >= 0
    assert plan.crop_x + w <= plan.exp_w
    assert plan.crop_y + h <= plan.exp_h


def test_plan_region_at_a_fractional_scale_takes_the_whole_output():
    """1.5 cannot place a region exactly, so capture everything and crop.

    wlroots computes the frame's device origin as ``trunc(lx * scale)``
    with a float scale we can only bound from above, so any origin but
    zero could be off by a pixel. A full-output capture is
    ``mode_w x mode_h`` by definition — no scale arithmetic, no guess.
    """
    plan = _plan_region(60, 30, 20, 10, 1920, 1080, 1280, 720)
    assert plan.region is None, "None means capture_output, not a region"
    assert (plan.crop_x, plan.crop_y) == (60, 30)
    assert (plan.exp_w, plan.exp_h) == (1920, 1080)


def test_plan_region_takes_the_whole_output_when_the_axes_disagree():
    plan = _plan_region(10, 10, 5, 5, 1920, 1080, 960, 1080)
    assert plan.region is None
    assert (plan.crop_x, plan.crop_y) == (10, 10)


# -------- what the backend refuses, and what it no longer refuses --------


@pytest.mark.parametrize("output, why", [
    (_output(transform=1), "a 90 degree rotation transposes the frame"),
    (_output(transform=3), "270 degrees likewise"),
    (_output(transform=4), "a flip reorders it"),
    (_scaled_output(200, 100, 100, 50, scale=2, transform=1), "both at once"),
])
def test_subregion_capture_is_refused_on_a_transformed_output(output, why):
    """wlroots applies the transform *before* the scale.

    So a rotation breaks the device/logical correspondence even on an
    unscaled output — a 20x10 request comes back 10x20 — and this
    backend does not model transforms. Refusing beats returning a
    region taken from a transposed backing store.
    """
    backend, _ = _fake_backend([output])
    with pytest.raises(NotImplementedError, match="logical coordinates"):
        backend.screenshot(0, 0, numpy.zeros((10, 20, 4), numpy.uint8))


def test_subregion_capture_is_allowed_on_an_identity_output():
    """The guard must not block the case that always worked."""
    backend, _ = _fake_backend([_output()])
    # No _screencopy on the fake, so reaching the capture call raises
    # AttributeError — which proves the guard did not fire.
    with pytest.raises(AttributeError):
        backend.screenshot(0, 0, numpy.zeros((10, 20, 4), numpy.uint8))


def test_subregion_capture_is_no_longer_refused_at_scale_2():
    """The refusal issue #38 tracks is lifted for a known integer scale."""
    output = _scaled_output(200, 100, 100, 50, scale=2)
    backend, _ = _fake_backend([output])
    with pytest.raises(AttributeError):
        backend.screenshot(20, 10, numpy.zeros((10, 20, 4), numpy.uint8))


@pytest.mark.parametrize("output", [
    _output(scale=2), _output(transform=1),
])
def test_full_output_capture_is_never_blocked_by_the_guard(output):
    """capture_output takes no region, so scale and transform cannot bite."""
    backend, _ = _fake_backend([output])
    with pytest.raises(AttributeError):
        backend.screenshot(0, 0, numpy.zeros((50, 100, 4), numpy.uint8))


class _FakeFrame:
    """A zwlr_screencopy_frame_v1 that reports a fixed buffer size."""

    def __init__(self, w, h, pixels=None, flags=0, fmts=(0,)):
        self.dispatcher = {}
        self.fmts = fmts
        self.w, self.h = w, h
        self.pixels = pixels
        self.flags = flags
        self.destroyed = False
        self.copied = False

    def deliver_buffer(self):
        # A version-3 compositor sends one buffer event per supported
        # format, then buffer_done. wl_shm ARGB8888 is 0, XRGB8888 is 1.
        for fmt in self.fmts:
            self.dispatcher["buffer"](self, fmt, self.w, self.h, self.w * 4)
        if self.flags:
            self.dispatcher["flags"](self, self.flags)
        self.dispatcher["buffer_done"](self)

    def deliver_ready(self):
        self.dispatcher["ready"](self, 0, 0, 0)

    def copy(self, wl_buffer):
        self.copied = True

    def destroy(self):
        self.destroyed = True


def _capture_backend(output, frame_w, frame_h):
    """A backend faked just far enough to reach the frame-size check."""
    backend, _ = _fake_backend([output])
    frame = _FakeFrame(frame_w, frame_h)
    backend._display = SimpleNamespace(
        dispatch=lambda block=True: frame.deliver_buffer(),
        roundtrip=lambda: None,
    )
    backend._screencopy = SimpleNamespace(
        capture_output=lambda overlay, out: frame,
        capture_output_region=lambda overlay, out, x, y, w, h: frame,
    )
    return backend, frame


def test_frame_smaller_than_the_request_is_refused_before_the_copy():
    """A fractionally scaled output cannot be caught by the up-front guard.

    wl_output.scale is an integer event and wlroots reports ceil() of
    the real scale, so an output at 0.75 announces scale 1 and looks
    like identity. The compositor returns a smaller frame, and
    ``img[:] = arr`` would *broadcast* a 1x1 frame across the 2x2
    destination rather than failing — silently wrong pixels.
    """
    backend, frame = _capture_backend(_output(), frame_w=1, frame_h=1)
    with pytest.raises(NotImplementedError, match="returned a 1x1 frame"):
        backend.screenshot(0, 0, numpy.zeros((2, 2, 4), numpy.uint8))
    assert frame.destroyed, "the frame must be released on the error path"


def test_frame_size_is_checked_for_full_output_capture_too():
    """The full-output path predicts the size exactly, so it can check it."""
    backend, frame = _capture_backend(_output(), frame_w=50, frame_h=25)
    with pytest.raises(NotImplementedError, match="returned a 50x25 frame"):
        backend.screenshot(0, 0, numpy.zeros((50, 100, 4), numpy.uint8))
    assert frame.destroyed


def test_matching_frame_size_passes_the_check():
    """The guard must not fire when the compositor returns what was asked."""
    backend, _ = _capture_backend(_output(), frame_w=2, frame_h=2)
    # Passing the size check, the capture goes on to _ensure_buffer,
    # which needs a real wl_shm pool — reaching it proves the check
    # allowed this through.
    with pytest.raises(AttributeError):
        backend.screenshot(0, 0, numpy.zeros((2, 2, 4), numpy.uint8))


def test_a_short_frame_on_the_scale_2_path_is_refused():
    """The size check is what *proves* the derived integer scale.

    xdg_output's logical size only bounds the real scale from above, so
    an output at 1.999 on a mode whose logical size happens to divide
    evenly would look like a clean scale of 2. It returns a frame one
    pixel short of the prediction, and that is the tell.
    """
    output = _scaled_output(200, 100, 100, 50, scale=2)
    backend, frame = _capture_backend(output, frame_w=19, frame_h=10)
    with pytest.raises(NotImplementedError, match="returned a 19x10 frame"):
        backend.screenshot(20, 10, numpy.zeros((10, 20, 4), numpy.uint8))
    assert frame.destroyed


# -------- the round trip: request, then crop what comes back --------


def _identifiable(w, h):
    """A frame whose every pixel encodes its own position in the frame."""
    ys, xs = numpy.mgrid[0:h, 0:w]
    pixels = numpy.zeros((h, w, 4), numpy.uint8)
    pixels[:, :, 0] = xs % 251   # B carries the column
    pixels[:, :, 1] = ys % 241   # G carries the row
    pixels[:, :, 2] = 0xAA
    pixels[:, :, 3] = 0xFF
    return pixels


def _pixel_backend(output, frame_w, frame_h, y_invert=False):
    """A backend faked all the way through the copy, with real pixels.

    Returns the backend, the frame content as the compositor's *upright*
    view of it, and a list that records the capture request.
    """
    backend, _ = _fake_backend([output])
    upright = _identifiable(frame_w, frame_h)
    # Y_INVERT says the buffer arrives bottom-up; the backend flips it
    # back, so hand it the flipped bytes to end up with `upright`.
    delivered = upright[::-1] if y_invert else upright
    frame = _FakeFrame(frame_w, frame_h, delivered,
                       flags=0x1 if y_invert else 0)

    stage = []

    def dispatch(block=True):
        if not stage:
            stage.append("buffer")
            frame.deliver_buffer()
        else:
            frame.deliver_ready()

    backend._display = SimpleNamespace(dispatch=dispatch, roundtrip=lambda: None)

    requests = []

    def capture_output(overlay, out):
        requests.append(("full",))
        return frame

    def capture_output_region(overlay, out, x, y, w, h):
        requests.append(("region", x, y, w, h))
        return frame

    backend._screencopy = SimpleNamespace(
        capture_output=capture_output,
        capture_output_region=capture_output_region,
    )
    # Stand in for the SHM pool: the backend only ever slices and copies.
    backend._ensure_buffer = lambda fmt, w, h, stride: (
        object(), numpy.ascontiguousarray(delivered).tobytes()
    )
    return backend, upright, requests


def _assert_is_frame_window(img, upright, x0, y0):
    """``img`` must be exactly the window of ``upright`` at ``(x0, y0)``."""
    h, w, _ = img.shape
    numpy.testing.assert_array_equal(img, upright[y0:y0 + h, x0:x0 + w])


def test_scale_2_subregion_is_requested_logical_and_cropped_back():
    """The whole point of the fix, end to end over a fake compositor."""
    output = _scaled_output(200, 100, 100, 50, scale=2)
    # logical (10, 5, 10, 5) -> a 20x10 device frame, no surplus
    backend, upright, requests = _pixel_backend(output, 20, 10)

    img = numpy.zeros((10, 20, 4), numpy.uint8)
    backend.screenshot(20, 10, img)

    assert requests == [("region", 10, 5, 10, 5)]
    _assert_is_frame_window(img, upright, 0, 0)


def test_scale_2_subregion_crops_off_the_outward_rounding():
    """An unaligned bbox rounds outward, then the surplus is cropped.

    The device bbox is (21, 11, 20, 10); the logical region rounds out
    to (10, 5, 11, 6), which comes back as a 22x12 frame whose top-left
    device pixel is (20, 10). The answer is the 20x10 window one pixel
    in from that corner.
    """
    output = _scaled_output(200, 100, 100, 50, scale=2)
    backend, upright, requests = _pixel_backend(output, 22, 12)

    img = numpy.zeros((10, 20, 4), numpy.uint8)
    backend.screenshot(21, 11, img)

    assert requests == [("region", 10, 5, 11, 6)]
    _assert_is_frame_window(img, upright, 1, 1)


def test_fractional_scale_subregion_crops_out_of_the_full_output():
    """1.5 falls back to a full-output capture and an exact crop."""
    # 30x15 device over 20x10 logical is a scale of 1.5, which wlroots
    # would announce as ceil() == 2 on wl_output.scale.
    output = _scaled_output(30, 15, 20, 10, scale=2)
    backend, upright, requests = _pixel_backend(output, 30, 15)

    img = numpy.zeros((5, 10, 4), numpy.uint8)
    backend.screenshot(7, 3, img)

    assert requests == [("full",)]
    _assert_is_frame_window(img, upright, 7, 3)


def test_unknown_logical_size_at_scale_2_crops_out_of_the_full_output():
    """No xdg_output: the integer scale is an upper bound, so take it all."""
    output = _output(mode_w=200, mode_h=100, scale=2)  # no logical size
    backend, upright, requests = _pixel_backend(output, 200, 100)

    img = numpy.zeros((10, 20, 4), numpy.uint8)
    backend.screenshot(60, 30, img)

    assert requests == [("full",)]
    _assert_is_frame_window(img, upright, 60, 30)


def test_y_invert_is_undone_before_the_crop():
    """The flag describes the frame, so flip first and crop second."""
    output = _scaled_output(200, 100, 100, 50, scale=2)
    backend, upright, _ = _pixel_backend(output, 22, 12, y_invert=True)

    img = numpy.zeros((10, 20, 4), numpy.uint8)
    backend.screenshot(21, 11, img)

    _assert_is_frame_window(img, upright, 1, 1)


def test_full_output_capture_still_copies_the_whole_frame():
    """The fast path is untouched: no region, no crop."""
    backend, upright, requests = _pixel_backend(_output(), 100, 50)

    img = numpy.zeros((50, 100, 4), numpy.uint8)
    backend.screenshot(0, 0, img)

    assert requests == [("full",)]
    _assert_is_frame_window(img, upright, 0, 0)


# -------- buffer format negotiation --------
#
# _on_frame_buffer used to begin `if state.fmt is None or fmt in (...)`,
# so whatever arrived *first* was taken whether or not its bytes were
# BGRA. ABGR8888 frames were copied out as if they were BGRA, turning
# red pixels blue.

_ABGR8888 = 0x34324241  # DRM fourcc 'AB24'; bytes are R, G, B, A
_XRGB8888 = 1
_ARGB8888 = 0


def test_a_format_that_is_not_bgra_is_refused_by_name():
    backend, frame = _capture_backend(_output(), frame_w=100, frame_h=50)
    frame.fmts = (_ABGR8888,)
    with pytest.raises(RuntimeError, match=r"no BGRA-compatible buffer format"):
        backend.screenshot(0, 0, numpy.zeros((50, 100, 4), numpy.uint8))
    assert frame.destroyed, "the frame must be released on the error path"


def test_the_refusal_names_the_format_that_was_offered():
    backend, frame = _capture_backend(_output(), frame_w=100, frame_h=50)
    frame.fmts = (_ABGR8888,)
    with pytest.raises(RuntimeError, match=r"0x34324241"):
        backend.screenshot(0, 0, numpy.zeros((50, 100, 4), numpy.uint8))


def _state_after(*fmts):
    """Feed buffer events to a fresh frame state, as a compositor would."""
    from fastgrab.backends.wlr import _FrameState

    state = _FrameState()
    for fmt in fmts:
        WlrBackend._on_frame_buffer(state, fmt, 100, 50, 400)
    return state


def test_a_non_bgra_format_is_never_chosen():
    state = _state_after(_ABGR8888)
    assert state.fmt is None, "ABGR8888 bytes are R,G,B,A — not the contract"
    assert state.offered == [_ABGR8888]


def test_an_acceptable_format_is_taken_even_if_offered_second():
    # The realistic shape: a v3 compositor advertising several types.
    state = _state_after(_ABGR8888, _XRGB8888)
    assert state.fmt == _XRGB8888
    assert state.offered == [_ABGR8888]


def test_the_first_acceptable_format_wins():
    state = _state_after(_ARGB8888, _XRGB8888)
    assert state.fmt == _ARGB8888


@pytest.mark.parametrize("fmt", [_ARGB8888, _XRGB8888])
def test_both_bgra_formats_are_accepted(fmt):
    state = _state_after(fmt)
    assert state.fmt == fmt
    assert state.stride == 400


# -------- screencopy version negotiation --------


class _FakeRegistry:
    def __init__(self, advertised):
        self.dispatcher = {}
        self._advertised = advertised
        self._emitted = False

    def emit(self):
        if self._emitted:
            return
        self._emitted = True
        for name, (interface, version) in enumerate(self._advertised):
            self.dispatcher["global"](self, name, interface, version)

    def bind(self, name, cls, version):
        return SimpleNamespace(dispatcher={})


def _connect_against(advertised, monkeypatch):
    """Drive _connect_singleton over a registry advertising `advertised`."""
    from fastgrab.backends import wlr as wlr_mod

    registry = _FakeRegistry(advertised)

    class _FakeDisplay:
        def connect(self):
            pass

        def get_registry(self):
            return registry

        def roundtrip(self):
            registry.emit()

    monkeypatch.setattr(wlr_mod, "Display", lambda *a, **kw: _FakeDisplay())
    # The connection is a process-wide singleton; do not leave ours behind.
    monkeypatch.setattr(wlr_mod, "_SINGLETON_STATE", None, raising=False)
    return wlr_mod.WlrBackend._connect_singleton()


def test_a_pre_v3_screencopy_compositor_is_refused_not_hung(monkeypatch):
    """Binding below version 3 would block forever, not fail.

    The capture loop waits for ``buffer_done``, which the protocol only
    added in version 3. A v1/v2 compositor sends ``buffer`` and then
    waits for ``copy`` while we wait for an event it will never send.
    """
    with pytest.raises(RuntimeError, match=r"version 2.*needs version 3"):
        _connect_against(
            [("zwlr_screencopy_manager_v1", 2), ("wl_shm", 1)], monkeypatch
        )


def test_the_pre_v3_refusal_explains_buffer_done(monkeypatch):
    with pytest.raises(RuntimeError, match=r"buffer_done"):
        _connect_against(
            [("zwlr_screencopy_manager_v1", 1), ("wl_shm", 1)], monkeypatch
        )


# -------- _ShmBuffer: freeing a frame buffer --------
#
# Nothing ever freed the per-instance SHM buffer, so each Screenshot
# leaked a memfd and a frame's worth of memory. The descriptor half is
# measured against a real compositor in tests/test_integration_wlr.py.
# The memory half is asserted here, because it is not ours to free: the
# compositor holds its own mapping of the memfd for as long as it owns
# the wl_buffer. Measured on cage, 50 dropped buffers held 175.5 MiB of
# system Shmem and closing every descriptor by hand released *none* of
# it; the destroy() requests took the same 50 down to 0.8 MiB. So the
# call that matters is destroy(), and these pin it down without reading
# a system-wide counter that CI cannot hold still.


class _FakeWlBuffer:
    """A wl_buffer proxy that records whether it was destroyed."""

    def __init__(self, fail=False):
        self.destroys = 0
        self._fail = fail

    def destroy(self):
        self.destroys += 1
        if self._fail:
            raise RuntimeError("connection is gone")


def _shm_buffer(**kw):
    """A real memfd + mmap, so close()/fd handling is not itself faked."""
    fd = os.memfd_create("fastgrab-wlr-test", 0)
    os.ftruncate(fd, 4096)
    mm = mmap.mmap(fd, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                   flags=mmap.MAP_SHARED)
    wl = _FakeWlBuffer(**kw)
    return _ShmBuffer(mm, fd, wl, 32, 32, 128, 0), mm, fd, wl


def _fd_is_open(fd):
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def test_closing_a_buffer_destroys_the_wl_buffer():
    """The one call that gets the memory back from the compositor."""
    buf, mm, fd, wl = _shm_buffer()
    buf.close()
    assert wl.destroys == 1
    assert mm.closed
    assert not _fd_is_open(fd)


def test_dropping_a_buffer_destroys_the_wl_buffer():
    """No close() anywhere — collection alone must free it.

    This is what removes the leak: WlrBackend has no __del__ and must not
    grow one, so the buffer has to free itself when the backend that
    owned it goes away.
    """
    buf, mm, fd, wl = _shm_buffer()
    del buf
    gc.collect()
    assert wl.destroys == 1
    assert mm.closed
    assert not _fd_is_open(fd)


def test_closing_a_buffer_twice_destroys_once():
    buf, _, _, wl = _shm_buffer()
    buf.close()
    buf.close()
    assert wl.destroys == 1


def test_an_explicitly_closed_buffer_is_not_freed_again_on_collection():
    buf, _, fd, wl = _shm_buffer()
    buf.close()
    del buf
    gc.collect()
    assert wl.destroys == 1
    assert not _fd_is_open(fd)


def test_the_finalizer_does_not_run_at_interpreter_exit():
    """Deliberate. Firing at shutdown is what makes __del__ unsafe here.

    destroy() is a request on the display connection, and at shutdown
    that connection may already be half torn down — the segfault the
    backend's "intentionally no __del__" note is about. The kernel
    reclaims the fd and the mapping then anyway, and the compositor
    drops everything when the socket closes, so there is nothing to win.
    """
    buf, _, _, _ = _shm_buffer()
    assert buf._finalize.atexit is False
    buf.close()


def test_a_dead_connection_does_not_raise_out_of_teardown():
    """destroy() on a lost connection is not a leak, and must not throw.

    _release_shm runs from a finalizer, where an exception surfaces at an
    arbitrary point in someone else's stack. The compositor frees every
    resource it held for us when the socket closes.
    """
    buf, mm, fd, wl = _shm_buffer(fail=True)
    buf.close()  # must not raise
    assert wl.destroys == 1
    # And our own handles still came back despite the failure above.
    assert mm.closed
    assert not _fd_is_open(fd)


def test_release_frees_our_handles_even_if_destroy_raises():
    """_release_shm directly, since the ordering is the whole point."""
    fd = os.memfd_create("fastgrab-wlr-test", 0)
    os.ftruncate(fd, 4096)
    mm = mmap.mmap(fd, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                   flags=mmap.MAP_SHARED)
    wl = _FakeWlBuffer(fail=True)
    _release_shm(mm, fd, wl)
    assert wl.destroys == 1
    assert mm.closed
    assert not _fd_is_open(fd)


def test_matches_distinguishes_every_dimension():
    """_ensure_buffer reuses on this; a loose compare would reuse wrongly."""
    buf, _, _, _ = _shm_buffer()
    assert buf.matches(32, 32, 128, 0)
    assert not buf.matches(33, 32, 128, 0)
    assert not buf.matches(32, 33, 128, 0)
    assert not buf.matches(32, 32, 129, 0)
    assert not buf.matches(32, 32, 128, 1)
    buf.close()
