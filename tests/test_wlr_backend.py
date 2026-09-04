"""Fake-state tests for the wlr backend that need no compositor.

``docker compose run --rm test-wayland`` exercises the real thing under
headless ``cage``, but that session has a single output and never
changes mode, so the paths :meth:`WlrBackend.refresh` exists for are
unreachable there. These drive the same code over hand-built output
state instead.
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
    output = SimpleNamespace(name="HDMI-1", mode_w=100, mode_h=50)

    def new_mode_arrives():
        output.mode_w, output.mode_h = 200, 120

    backend, roundtrips = _fake_backend([output], new_mode_arrives)
    assert backend.resolution() == (100, 50)

    backend.refresh()
    assert backend.resolution() == (200, 120)
    # two, matching what _connect_singleton does to let events settle
    assert len(roundtrips) == 2


def test_refresh_reselects_the_requested_output(monkeypatch):
    first = SimpleNamespace(name="HDMI-1", mode_w=100, mode_h=50)
    second = SimpleNamespace(name="DP-1", mode_w=300, mode_h=150)
    backend, _ = _fake_backend([first, second])
    assert backend._output is first

    monkeypatch.setenv("FASTGRAB_OUTPUT", "DP-1")
    backend.refresh()
    assert backend._output is second
    assert backend.resolution() == (300, 150)


def test_subregion_capture_is_refused_on_a_scaled_output():
    """Device pixels and wlr-screencopy logical coordinates agree at 1x only.

    The protocol takes the region in logical coordinates, so a 20x10
    device-pixel request on a scale-2 output would be sent as 20x10
    logical, come back as a 40x20 frame, and be displaced at the origin
    too. Refusing beats returning the wrong pixels.
    """
    output = SimpleNamespace(name="HDMI-1", mode_w=100, mode_h=50, scale=2)
    backend, _ = _fake_backend([output])
    with pytest.raises(NotImplementedError, match="logical coordinates"):
        backend.screenshot(0, 0, numpy.zeros((10, 20, 4), numpy.uint8))


def test_full_output_capture_is_not_blocked_by_scale():
    """Full-output capture goes via capture_output and takes no region."""
    output = SimpleNamespace(name="HDMI-1", mode_w=100, mode_h=50, scale=2)
    backend, _ = _fake_backend([output])
    # No _screencopy on the fake, so reaching the capture call raises
    # AttributeError -- which proves the scale guard did not fire.
    with pytest.raises(AttributeError):
        backend.screenshot(0, 0, numpy.zeros((50, 100, 4), numpy.uint8))


def test_refresh_raises_for_an_unknown_requested_output(monkeypatch):
    output = SimpleNamespace(name="HDMI-1", mode_w=100, mode_h=50)
    backend, _ = _fake_backend([output])

    monkeypatch.setenv("FASTGRAB_OUTPUT", "DP-9")
    with pytest.raises(RuntimeError, match="not found"):
        backend.refresh()
