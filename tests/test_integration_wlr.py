"""Pixel-level tests for the Wayland (wlr-screencopy) backend.

Run inside the ``test-wayland`` docker compose service, which boots
``cage`` headlessly with ``tests/wayland_painter.py`` as its kiosk client.
The :func:`paint_wayland` fixture in :mod:`tests.conftest` drives the
painter over a UNIX socket; we capture via :class:`fastgrab.WlrBackend`
and assert on the resulting BGRA bytes.

These tests are tagged with the ``wayland`` marker so they're excluded
from the X11 test run and selected explicitly via ``-m wayland``.
"""
import gc
import os
import time

import numpy
import pytest

from fastgrab import screenshot
from fastgrab.backends.wlr import WlrBackend, _uniform_integer_scale

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


# -------- sub-region placement (issue #38) --------
#
# A solid color proves a capture has the right shape and nothing about
# where it came from. These use the painter's `blocks` pattern, whose
# every 8x8 block encodes its own coordinates, and compare a sub-region
# against the same window of a full-output capture. Full-output capture
# goes through `capture_output`, which takes no region at all, so it is
# unaffected by the logical/device mix-up and can serve as the oracle.


def _stable_capture(g, bbox=None, tries=20):
    """Capture until two consecutive frames agree.

    Changing the output scale makes cage reconfigure its kiosk client,
    and a capture taken while that is still in flight sees a
    half-updated scene. These tests compare two captures against each
    other, so both have to be of the settled scene.
    """
    previous = g.capture(bbox=bbox).copy()
    for _ in range(tries):
        time.sleep(0.05)
        current = g.capture(bbox=bbox).copy()
        if numpy.array_equal(previous, current):
            return current
        previous = current
    raise AssertionError("the compositor never settled on a stable frame")


def _assert_subregion_matches_full_output(g, bbox):
    full = _stable_capture(g)
    assert len(numpy.unique(full[:, :, R])) > 4, (
        "the block pattern is what makes a displaced capture detectable; "
        "against a flat frame this assertion would pass vacuously"
    )

    x, y, w, h = bbox
    sub = _stable_capture(g, bbox=bbox)
    assert sub.shape == (h, w, 4)
    numpy.testing.assert_array_equal(sub, full[y:y + h, x:x + w])
    if (x, y) != (0, 0):
        assert not numpy.array_equal(sub, full[0:h, 0:w]), (
            "the pattern must differ between the bbox and the output "
            "origin, or a capture ignoring the origin would still pass"
        )


_PLACEMENT_BBOXES = [
    (0, 0, 64, 48),      # at the origin, aligned to any integer scale
    (21, 11, 20, 10),    # odd origin — the logical region rounds outward
    (128, 64, 100, 60),  # aligned, well away from the origin
    (301, 177, 33, 21),  # odd origin and odd size
]


@pytest.mark.parametrize("bbox", _PLACEMENT_BBOXES)
def test_wlr_subregion_comes_from_the_right_place(paint_wayland, bbox):
    """The unscaled control for the scale-2 test below."""
    paint_wayland("blocks")
    _assert_subregion_matches_full_output(_grab(), bbox)


def test_wlr_tracks_the_output_logical_size():
    """xdg_output is what grounds the device -> logical conversion.

    If ``zxdg_output_manager_v1`` were not bound, or the second-stage
    ``get_xdg_output`` call were made before the registry had announced
    the manager, this would silently stay 0 and every scaled sub-region
    would quietly fall back to a full-output capture.
    """
    backend = WlrBackend()
    backend.refresh()
    output = backend._output
    assert output.logical_w > 0 and output.logical_h > 0
    assert (output.logical_w, output.logical_h) == backend.resolution(), (
        "cage's headless output is unscaled, so logical == device here"
    )


@pytest.mark.parametrize("bbox", _PLACEMENT_BBOXES)
@pytest.mark.parametrize("scale, exact", [
    # cage's headless output is 1280x720. At 2 the logical size is a
    # clean 640x360, so the backend converts the bbox and asks for a
    # logical region. At 1.5 it is 853x480 — 1280 is not a whole
    # multiple of 853, the scale cannot be proved, and the backend falls
    # back to capturing the whole output and cropping.
    (2, 2),
    (1.5, None),
])
def test_wlr_subregion_comes_from_the_right_place_when_scaled(
        paint_wayland, wlr_output_scale, scale, exact, bbox):
    """Issue #38: the region is logical while the bbox is device pixels.

    On a scale-2 output a 20x10 device request must go out as 10x5
    logical from a halved origin. Passing the bbox through unconverted,
    as this backend used to, asks for twice the region from twice the
    offset — the compositor returns a 40x20 frame that does not fit the
    destination, and such pixels as do arrive are from the wrong place.
    """
    wlr_output_scale(scale)
    paint_wayland("blocks")

    g = _grab()
    output = g._backend._output
    assert 0 < output.logical_w < g.screensize[0], (
        "expected wlr-randr to have scaled the output, got mode {} "
        "logical {}".format(g.screensize,
                            (output.logical_w, output.logical_h))
    )
    assert _uniform_integer_scale(
        output.mode_w, output.mode_h, output.logical_w, output.logical_h
    ) == exact, "this parametrization is meant to exercise the other branch"

    _assert_subregion_matches_full_output(g, bbox)


# -------- frame-buffer lifetime --------
#
# Each WlrBackend allocates a memfd-backed wl_buffer and nothing ever
# freed it, so every Screenshot instance leaked one descriptor and one
# frame's worth of memory. Measured before the fix, against cage: 76
# leaked memfds after 76 instances, and with RLIMIT_NOFILE lowered to 64
# the 43rd capture died with "OSError: [Errno 24] Too many open files".
#
# The memory is the worse half and is *not* ours to free directly. The
# compositor keeps its own mapping for as long as it owns the wl_buffer,
# so closing our descriptor releases nothing — 50 dropped buffers held
# 175.5 MiB of system Shmem, and closing every fd by hand freed none of
# it; the wl_buffer.destroy() requests brought it to 0.8 MiB. That is
# asserted deterministically in tests/test_wlr_backend.py rather than
# here, where it would mean reading a system-wide counter.

# One live frame buffer accounts for *two* descriptors, not one:
# os.memfd_create gives us ours, and mmap.mmap() dups it internally and
# holds that copy until mm.close(). Both are freed together. Before the
# fix, GC reclaimed the mmap (and so its dup) while the memfd itself was
# never closed, which is why the leak measured one descriptor per
# instance rather than two.
_FDS_PER_BUFFER = 2


def _fastgrab_memfds():
    """Count this process's open ``fastgrab-wlr`` descriptors.

    Collects first. Most tests in this module drop their backend without
    closing it — the normal way to use the library — so an uncollected
    buffer from an earlier test would otherwise sit in the baseline and
    make these counts depend on test order.
    """
    gc.collect()
    total = 0
    for entry in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink("/proc/self/fd/" + entry)
        except OSError:
            continue  # the descriptor closed under us; it is not a leak
        if "fastgrab-wlr" in target:
            total += 1
    return total


def test_a_capture_opens_one_frame_buffer():
    """Baseline for the leak tests: one buffer per live backend."""
    before = _fastgrab_memfds()
    with _grab() as grab:
        grab.capture((0, 0, 32, 32))
        assert _fastgrab_memfds() == before + _FDS_PER_BUFFER
    assert _fastgrab_memfds() == before


def test_close_releases_the_frame_buffer_immediately():
    before = _fastgrab_memfds()
    grab = _grab()
    try:
        grab.capture((0, 0, 32, 32))
    finally:
        grab.close()
    assert _fastgrab_memfds() == before


def test_dropped_screenshot_objects_do_not_leak():
    """The leak as a user meets it: no close(), just instances going away.

    This is the shape of the library's headline API — the object in
    ``Screenshot().capture()`` is unreachable the moment the expression
    ends — so collection alone has to be enough.
    """
    before = _fastgrab_memfds()
    for _ in range(20):
        grab = _grab()
        grab.capture((0, 0, 32, 32))
        del grab
    gc.collect()
    assert _fastgrab_memfds() == before


def test_the_two_line_api_in_a_loop_does_not_leak():
    """Verbatim the form the README advertises, thirty times over."""
    before = _fastgrab_memfds()
    for _ in range(30):
        screenshot.Screenshot(backend="wlr").capture((0, 0, 16, 16))
    gc.collect()
    assert _fastgrab_memfds() == before


def test_resizing_within_one_instance_reuses_one_buffer():
    """The resize path already freed correctly; keep it that way."""
    before = _fastgrab_memfds()
    with _grab() as grab:
        for size in (16, 32, 64, 32, 16):
            grab.capture((0, 0, size, size))
            assert _fastgrab_memfds() == before + _FDS_PER_BUFFER
    assert _fastgrab_memfds() == before


def test_a_closed_instance_does_not_disturb_a_live_one():
    """The display is a process-wide singleton; the buffer is not."""
    first, second = _grab(), _grab()
    try:
        first.capture((0, 0, 32, 32))
        second.capture((0, 0, 32, 32))
        first.close()
        # The survivor keeps working, and still owns exactly one buffer.
        assert second.capture((0, 0, 32, 32)).shape == (32, 32, 4)
    finally:
        first.close()
        second.close()
