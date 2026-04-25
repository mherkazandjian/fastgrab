"""End-to-end tests that paint the X root window and verify capture content.

These tests need a running X server (typically Xvfb under
``docker compose run --rm test``) and the ``xsetroot`` command, which is
provided by the ``x11-xserver-utils`` package. They are tagged with the
``integration`` marker so they can be deselected with ``-m 'not integration'``.
"""
import numpy
import pytest

from fastgrab import screenshot

pytestmark = pytest.mark.integration


# X11 ZPixmap on little-endian Linux x86_64 lays out each pixel as B, G, R, A.
B, G, R, A = 0, 1, 2, 3


def test_capture_solid_red_fills_red_channel(paint_root):
    paint_root("#FF0000")
    img = screenshot.Screenshot().capture()
    assert (img[:, :, R] == 255).all()
    assert (img[:, :, G] == 0).all()
    assert (img[:, :, B] == 0).all()


def test_capture_solid_green_fills_green_channel(paint_root):
    paint_root("#00FF00")
    img = screenshot.Screenshot().capture()
    assert (img[:, :, R] == 0).all()
    assert (img[:, :, G] == 255).all()
    assert (img[:, :, B] == 0).all()


def test_capture_solid_blue_fills_blue_channel(paint_root):
    paint_root("#0000FF")
    img = screenshot.Screenshot().capture()
    assert (img[:, :, R] == 0).all()
    assert (img[:, :, G] == 0).all()
    assert (img[:, :, B] == 255).all()


def test_capture_arbitrary_color_matches_bytes(paint_root):
    paint_root("#0A141E")  # R=10, G=20, B=30
    img = screenshot.Screenshot().capture()
    assert (img[:, :, R] == 10).all()
    assert (img[:, :, G] == 20).all()
    assert (img[:, :, B] == 30).all()


def test_capture_reflects_state_changes_between_calls(paint_root):
    grab = screenshot.Screenshot()
    paint_root("#FF0000")
    red = grab.capture().copy()
    paint_root("#00FF00")
    green = grab.capture().copy()
    assert not numpy.array_equal(red, green)
    assert (red[:, :, R] == 255).all()
    assert (green[:, :, G] == 255).all()


def test_capture_subregion_matches_solid_color(paint_root):
    paint_root("#FF00FF")  # R=255, G=0, B=255
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    bw, bh = min(64, w), min(48, h)
    img = grab.capture(bbox=(0, 0, bw, bh))
    assert img.shape == (bh, bw, 4)
    assert (img[:, :, R] == 255).all()
    assert (img[:, :, G] == 0).all()
    assert (img[:, :, B] == 255).all()


def test_capture_subregion_with_offset_within_solid_field(paint_root):
    """When the whole root is one color, an offset bbox is still that color."""
    paint_root("#123456")  # R=18, G=52, B=86
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    bw, bh = min(32, w - 5), min(32, h - 5)
    img = grab.capture(bbox=(5, 5, bw, bh))
    assert img.shape == (bh, bw, 4)
    assert (img[:, :, R] == 0x12).all()
    assert (img[:, :, G] == 0x34).all()
    assert (img[:, :, B] == 0x56).all()
