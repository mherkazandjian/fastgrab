"""Pixel-level tests for the Wayland (wlr-screencopy) backend.

Run inside the ``test-wayland`` docker compose service, which boots
``cage`` headlessly with ``tests/wayland_painter.py`` as its kiosk client.
The :func:`paint_wayland` fixture in :mod:`tests.conftest` drives the
painter over a UNIX socket; we capture via :class:`fastgrab.WlrBackend`
and assert on the resulting BGRA bytes.

These tests are tagged with the ``wayland`` marker so they're excluded
from the X11 test run and selected explicitly via ``-m wayland``.
"""
import numpy
import pytest

from fastgrab import screenshot
from fastgrab.backends.wlr import WlrBackend

pytestmark = pytest.mark.wayland


# wlr-screencopy on cage gives BGRA byte order (B=0, G=1, R=2, A=3) — the
# same contract as the X11 backend on little-endian Linux x86_64.
B, G, R, A = 0, 1, 2, 3


def _grab():
    return screenshot.Screenshot(backend="wlr")


def test_wlr_backend_can_be_constructed():
    WlrBackend()


def test_wlr_dispatcher_picks_wayland():
    g = screenshot.Screenshot()
    assert isinstance(g._backend, WlrBackend)


def test_wlr_resolution_is_positive_2_tuple():
    g = _grab()
    w, h = g.screensize
    assert isinstance(w, int) and isinstance(h, int)
    assert w > 0 and h > 0


def test_wlr_capture_full_screen_shape_and_dtype(paint_wayland):
    paint_wayland("#000000")
    g = _grab()
    w, h = g.screensize
    img = g.capture()
    assert img.shape == (h, w, 4)
    assert img.dtype == numpy.uint8


def test_wlr_capture_solid_red(paint_wayland):
    paint_wayland("#FF0000")
    img = _grab().capture()
    assert (img[:, :, R] == 255).all()
    assert (img[:, :, G] == 0).all()
    assert (img[:, :, B] == 0).all()


def test_wlr_capture_solid_green(paint_wayland):
    paint_wayland("#00FF00")
    img = _grab().capture()
    assert (img[:, :, R] == 0).all()
    assert (img[:, :, G] == 255).all()
    assert (img[:, :, B] == 0).all()


def test_wlr_capture_solid_blue(paint_wayland):
    paint_wayland("#0000FF")
    img = _grab().capture()
    assert (img[:, :, R] == 0).all()
    assert (img[:, :, G] == 0).all()
    assert (img[:, :, B] == 255).all()


def test_wlr_capture_arbitrary_color(paint_wayland):
    paint_wayland("#0A141E")  # R=10, G=20, B=30
    img = _grab().capture()
    assert (img[:, :, R] == 10).all()
    assert (img[:, :, G] == 20).all()
    assert (img[:, :, B] == 30).all()


def test_wlr_capture_reflects_state_changes_between_calls(paint_wayland):
    g = _grab()
    paint_wayland("#FF0000")
    red = g.capture().copy()
    paint_wayland("#00FF00")
    green = g.capture().copy()
    assert not numpy.array_equal(red, green)
    assert (red[:, :, R] == 255).all()
    assert (green[:, :, G] == 255).all()


def test_wlr_capture_subregion_shape(paint_wayland):
    paint_wayland("#FF00FF")
    g = _grab()
    w, h = g.screensize
    bw, bh = min(64, w), min(48, h)
    img = g.capture(bbox=(0, 0, bw, bh))
    assert img.shape == (bh, bw, 4)
    assert (img[:, :, R] == 255).all()
    assert (img[:, :, G] == 0).all()
    assert (img[:, :, B] == 255).all()
