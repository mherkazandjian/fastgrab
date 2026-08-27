"""Unit tests for fastgrab.effects — pure numpy, no display needed.

These run everywhere, including the Windows and macOS CI runners: the
module only ever touches a numpy array handed to it.
"""
import numpy
import pytest

from fastgrab.effects import (
    BLUR_METHODS,
    BlurStyle,
    blur_regions,
)


# BGRA channel indices, same convention as tests/test_integration.py.
B, G, R, A = 0, 1, 2, 3

BLUR_ONLY = [m for m in BLUR_METHODS if m != "fill"]


def _noise(h, w, seed=0, alpha=255):
    """A random BGRA frame with a fixed, non-uniform alpha channel."""
    rng = numpy.random.default_rng(seed)
    img = numpy.empty((h, w, 4), numpy.uint8)
    img[..., :3] = rng.integers(0, 256, (h, w, 3), dtype=numpy.uint8)
    img[..., A] = alpha
    return img


# --------------------------------------------------------------------
# BlurStyle validation
# --------------------------------------------------------------------

def test_blur_style_defaults_are_a_box_blur():
    style = BlurStyle()
    assert style.method == "box"
    assert style.radius > 0
    assert style.color == (0, 0, 0)


def test_blur_style_rejects_unknown_method():
    with pytest.raises(ValueError):
        BlurStyle(method="swirl")


def test_blur_style_rejects_bad_numbers():
    with pytest.raises(ValueError):
        BlurStyle(radius=-1)
    with pytest.raises(ValueError):
        BlurStyle(block=0)
    with pytest.raises(ValueError):
        BlurStyle(passes=0)


def test_blur_style_rejects_bad_colour():
    with pytest.raises(ValueError):
        BlurStyle(method="fill", color=(0, 0))
    with pytest.raises(ValueError):
        BlurStyle(method="fill", color=(0, 0, 300))


def test_blur_style_normalises_colour_to_ints():
    assert BlurStyle(color=[1.0, 2.0, 3.0]).color == (1, 2, 3)


# --------------------------------------------------------------------
# What the modes do to the pixels
# --------------------------------------------------------------------

@pytest.mark.parametrize("method", BLUR_METHODS)
def test_constant_region_survives_every_method(method):
    """Averaging a flat field must give back the same flat field."""
    img = numpy.full((40, 60, 4), 77, numpy.uint8)
    blur_regions(img, None, BlurStyle(method=method, color=(77, 77, 77)))
    assert (img[..., :3] == 77).all()


@pytest.mark.parametrize("method", BLUR_METHODS)
def test_alpha_channel_is_never_touched(method):
    img = _noise(24, 32, alpha=123)
    blur_regions(img, [(4, 4, 16, 12)], BlurStyle(method=method))
    assert (img[..., A] == 123).all()


@pytest.mark.parametrize("method", BLUR_ONLY)
def test_blur_reduces_variance(method):
    img = _noise(48, 48, seed=1)
    before = img[..., :3].astype(float).var()
    blur_regions(img, None, BlurStyle(method=method, radius=6, block=8))
    assert img[..., :3].astype(float).var() < before / 4.0


def test_box_blur_spreads_an_impulse_without_leaking_energy():
    img = numpy.zeros((32, 32, 4), numpy.uint8)
    img[16, 16, B] = 255
    blur_regions(img, None, BlurStyle(method="box", radius=3))
    # A 7x7 window: the peak drops to ~255/49 and the neighbourhood lights up.
    assert 0 < img[16, 16, B] < 255
    assert (img[13:20, 13:20, B] > 0).all()
    assert (img[0:12, :, B] == 0).all()


def test_channels_do_not_mix():
    """A blue-only frame must stay blue-only through a blur."""
    img = numpy.zeros((32, 32, 4), numpy.uint8)
    img[8:24, 8:24, B] = 255
    blur_regions(img, None, BlurStyle(method="gaussian", radius=5))
    assert img[..., B].any()
    assert not img[..., G].any()
    assert not img[..., R].any()


def test_fill_writes_exactly_the_requested_bgr():
    img = numpy.full((20, 20, 4), 200, numpy.uint8)
    blur_regions(
        img, [(5, 6, 4, 3)], BlurStyle(method="fill", color=(10, 20, 30))
    )
    box = img[6:9, 5:9]
    assert (box[..., B] == 10).all()
    assert (box[..., G] == 20).all()
    assert (box[..., R] == 30).all()
    assert (img[0:5, :, :3] == 200).all()


def test_pixelate_makes_each_tile_uniform():
    img = _noise(16, 16, seed=2)
    blur_regions(img, None, BlurStyle(method="pixelate", block=4))
    for y in range(0, 16, 4):
        for x in range(0, 16, 4):
            tile = img[y:y + 4, x:x + 4, :3]
            assert (tile == tile[0, 0]).all()


def test_pixelate_handles_a_region_not_divisible_by_the_block():
    img = _noise(10, 7, seed=3)
    blur_regions(img, None, BlurStyle(method="pixelate", block=4))
    # The ragged 2-wide / 2-tall edge tiles are averaged too, not dropped.
    assert (img[8:10, 4:7, :3] == img[8, 4, :3]).all()


# --------------------------------------------------------------------
# Region handling
# --------------------------------------------------------------------

@pytest.mark.parametrize("method", BLUR_METHODS)
def test_pixels_outside_the_region_are_byte_identical(method):
    img = _noise(40, 40, seed=4)
    before = img.copy()
    blur_regions(img, [(10, 10, 20, 20)], BlurStyle(method=method))
    assert (img[0:10] == before[0:10]).all()
    assert (img[30:] == before[30:]).all()
    assert (img[:, 0:10] == before[:, 0:10]).all()
    assert (img[:, 30:] == before[:, 30:]).all()
    assert not (img[10:30, 10:30, :3] == before[10:30, 10:30, :3]).all()


def test_regions_none_blurs_the_whole_frame():
    img = _noise(20, 20, seed=5)
    before = img.copy()
    blur_regions(img, None, BlurStyle(method="fill", color=(1, 2, 3)))
    assert (img[..., B] == 1).all()
    assert not (img[..., :3] == before[..., :3]).all()


def test_empty_region_list_is_a_no_op():
    img = _noise(16, 16, seed=6)
    before = img.copy()
    blur_regions(img, [], BlurStyle(method="fill"))
    assert (img == before).all()


def test_regions_are_clipped_not_wrapped():
    """A negative or oversized rectangle must not blur the opposite edge."""
    img = numpy.full((10, 10, 4), 50, numpy.uint8)
    blur_regions(
        img,
        [(-5, -5, 3, 3),      # entirely off the top-left
         (100, 100, 5, 5),    # entirely off the bottom-right
         (8, 8, 20, 20)],     # overlaps, extends past the frame
        BlurStyle(method="fill", color=(1, 2, 3)),
    )
    assert (img[8:10, 8:10, B] == 1).all()
    assert (img[0:8, :, B] == 50).all()
    assert (img[:, 0:8, B] == 50).all()


def test_partially_offscreen_region_is_clipped_at_the_origin():
    img = numpy.full((10, 10, 4), 50, numpy.uint8)
    blur_regions(img, [(-2, -2, 5, 5)], BlurStyle(method="fill", color=(9, 9, 9)))
    assert (img[0:3, 0:3, B] == 9).all()
    assert (img[3:, 3:, B] == 50).all()


def test_zero_sized_regions_are_skipped():
    img = _noise(12, 12, seed=7)
    before = img.copy()
    blur_regions(img, [(2, 2, 0, 5), (2, 2, 5, 0)], BlurStyle(method="fill"))
    assert (img == before).all()


def test_origin_translates_screen_coordinates_into_the_frame():
    img = numpy.full((10, 10, 4), 50, numpy.uint8)
    blur_regions(
        img, [(105, 104, 2, 3)],
        BlurStyle(method="fill", color=(9, 9, 9)), origin=(100, 100),
    )
    assert (img[4:7, 5:7, B] == 9).all()
    assert img[0, 0, B] == 50


# --------------------------------------------------------------------
# No-ops, return value, scratch reuse
# --------------------------------------------------------------------

@pytest.mark.parametrize("style", [
    BlurStyle(method="box", radius=0),
    BlurStyle(method="gaussian", radius=0),
    BlurStyle(method="pixelate", block=1),
])
def test_degenerate_settings_leave_the_frame_alone(style):
    img = _noise(16, 16, seed=8)
    before = img.copy()
    blur_regions(img, None, style)
    assert (img == before).all()


def test_blur_regions_returns_the_same_array_object():
    img = _noise(8, 8, seed=9)
    assert blur_regions(img, None, BlurStyle(method="fill")) is img


def test_scratch_reuse_does_not_change_the_result():
    """The reused work buffers must be an optimisation, nothing more."""
    style = BlurStyle(method="gaussian", radius=4)
    regions = [(3, 3, 20, 14)]
    a = _noise(24, 30, seed=10)
    b = a.copy()
    scratch = {}
    # Prime the scratch dict with a different shape first, so the second
    # call exercises both reuse and a fresh allocation.
    blur_regions(_noise(12, 12, seed=11), None, style, scratch=scratch)
    blur_regions(a, regions, style, scratch=scratch)
    blur_regions(b, regions, style)
    assert (a == b).all()


def test_scratch_dict_stays_bounded():
    style = BlurStyle(radius=2)
    scratch = {}
    for size in range(4, 60):
        blur_regions(_noise(size, size, seed=size), None, style,
                     scratch=scratch)
    assert len(scratch) <= 32
