"""Fake-state tests for the wlr backend that need no compositor.

``docker compose run --rm test-wayland`` exercises the real thing under
headless ``cage``, but that session has a single unscaled, untransformed
output and never changes mode, so the paths :meth:`WlrBackend.refresh`
and the sub-region guard exist for are unreachable there. These drive
the same code over hand-built output state instead.
"""
from types import SimpleNamespace

import numpy
import pytest

pytest.importorskip(
    "pywayland", reason="the wlr backend is behind the [wayland] extra"
)

from fastgrab.backends.wlr import WlrBackend  # noqa: E402

# Hand-built output state only — no compositor, no display server.
pytestmark = pytest.mark.no_display


def _output(name="HDMI-1", mode_w=100, mode_h=50, scale=1, transform=0):
    """An output as ``_connect_singleton`` would have accumulated it."""
    return SimpleNamespace(proxy=object(), name=name, mode_w=mode_w,
                           mode_h=mode_h, scale=scale, transform=transform)


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


@pytest.mark.parametrize("output, why", [
    (_output(scale=2), "a scale of 2 doubles the frame"),
    (_output(transform=1), "a 90 degree rotation transposes it"),
    (_output(transform=3), "270 degrees likewise"),
    (_output(scale=2, transform=1), "both at once"),
])
def test_subregion_capture_is_refused_off_the_identity_mapping(output, why):
    """The region is logical; device pixels match only at scale 1, no transform.

    wlroots applies the output transform *before* the scale, so a
    rotation breaks the correspondence even on an unscaled output — a
    20x10 request comes back 10x20. Refusing beats returning the wrong
    pixels, which is what a scale-only guard would have done here.
    """
    backend, _ = _fake_backend([output])
    with pytest.raises(NotImplementedError, match="logical coordinates"):
        backend.screenshot(0, 0, numpy.zeros((10, 20, 4), numpy.uint8))


def test_subregion_capture_is_allowed_on_an_identity_output():
    """The guard must not block the case that actually works."""
    backend, _ = _fake_backend([_output()])
    # No _screencopy on the fake, so reaching the capture call raises
    # AttributeError — which proves the guard did not fire.
    with pytest.raises(AttributeError):
        backend.screenshot(0, 0, numpy.zeros((10, 20, 4), numpy.uint8))


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

    def __init__(self, w, h):
        self.dispatcher = {}
        self.w, self.h = w, h
        self.destroyed = False

    def deliver_buffer(self):
        # wl_shm ARGB8888 is format 0.
        self.dispatcher["buffer"](self, 0, self.w, self.h, self.w * 4)
        self.dispatcher["buffer_done"](self)

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
