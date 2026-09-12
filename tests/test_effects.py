"""Unit tests for fastgrab.effects — pure numpy, no display needed.

These run everywhere, including the Windows and macOS CI runners: the
module only ever touches a numpy array handed to it.
"""
import numpy
import pytest

from fastgrab.effects import (
    BLUR_METHODS,
    IMAGE_FITS,
    BlurStyle,
    blur_regions,
)


# BGRA channel indices, same convention as tests/test_integration.py.
B, G, R, A = 0, 1, 2, 3

# Modes whose output is a function of the style alone, not of the pixels
# underneath — which is the point of them, and why "a flat field
# survives" and "variance drops" are the wrong assertions there.
CONTENT_INDEPENDENT = ("fill", "pixelate-random", "image")

# The rest: modes derived from the pixels they cover.
AVERAGING = [m for m in BLUR_METHODS if m not in CONTENT_INDEPENDENT]


def _cover(h=4, w=4):
    """A small, unmistakable BGR cover image."""
    img = numpy.zeros((h, w, 3), numpy.uint8)
    img[..., 2] = 255                      # red in BGR
    img[0, 0] = (255, 0, 0)                # one blue corner, to catch flips
    return img


def _style(method, **kwargs):
    """A valid BlurStyle for ``method``, supplying whatever it requires."""
    if method == "image":
        kwargs.setdefault("image", _cover())
    return BlurStyle(method=method, **kwargs)


def _noise(h, w, seed=0, alpha=None):
    """A random BGRA frame.

    ``alpha=None`` fills the alpha channel with a varying pattern rather
    than a constant, so "the blur leaves alpha alone" is checked against
    something a stray write would actually disturb.
    """
    rng = numpy.random.default_rng(seed)
    img = numpy.empty((h, w, 4), numpy.uint8)
    img[..., :3] = rng.integers(0, 256, (h, w, 3), dtype=numpy.uint8)
    if alpha is None:
        img[..., A] = rng.integers(0, 256, (h, w), dtype=numpy.uint8)
    else:
        img[..., A] = alpha
    return img


def _reference_box(plane, radius):
    """Brute-force box blur of a 2-D uint8 plane, clamped at the edges.

    Deliberately written as slow nested loops over explicit window
    bounds: it shares no code with the cumsum implementation, so it can
    catch an off-by-one that a self-consistent test would not.
    """
    h, w = plane.shape
    src = plane.astype(numpy.float64)
    horizontal = numpy.empty((h, w), numpy.float64)
    for x in range(w):
        lo, hi = max(x - radius, 0), min(x + radius + 1, w)
        horizontal[:, x] = src[:, lo:hi].mean(axis=1)
    out = numpy.empty((h, w), numpy.float64)
    for y in range(h):
        lo, hi = max(y - radius, 0), min(y + radius + 1, h)
        out[y, :] = horizontal[lo:hi, :].mean(axis=0)
    return out


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

@pytest.mark.parametrize("method", AVERAGING + ["fill"])
def test_constant_region_survives_every_method(method):
    """Averaging a flat field must give back the same flat field.

    pixelate-random is excluded on purpose: it ignores the content, so a
    flat field does not survive it and should not.
    """
    img = numpy.full((40, 60, 4), 77, numpy.uint8)
    blur_regions(img, None, BlurStyle(method=method, color=(77, 77, 77)))
    assert (img[..., :3] == 77).all()


def test_a_constant_region_does_not_survive_pixelate_random():
    """The counterpart: content-independence means the flat field goes."""
    img = numpy.full((40, 60, 4), 77, numpy.uint8)
    blur_regions(img, None, BlurStyle(method="pixelate-random", block=8))
    assert not (img[..., :3] == 77).all()


@pytest.mark.parametrize("method", BLUR_METHODS)
def test_alpha_channel_is_never_touched(method):
    img = _noise(24, 32, seed=12)
    before = img[..., A].copy()
    blur_regions(img, [(4, 4, 16, 12)], _style(method))
    assert (img[..., A] == before).all()


@pytest.mark.parametrize("method", AVERAGING)
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
    blur_regions(img, [(10, 10, 20, 20)], _style(method))
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


@pytest.mark.parametrize("region", [
    (2, 2, 0, 5), (2, 2, 5, 0), (2, 2, -3, 5),
])
def test_zero_sized_regions_raise_rather_than_being_skipped(region):
    """An empty rectangle looks like a real target but covers nothing.

    Silently skipping it let blur_regions return successfully with the
    frame untouched — a redaction the caller believes happened.
    """
    img = _noise(12, 12, seed=7)
    with pytest.raises(ValueError, match="positive"):
        blur_regions(img, [region], BlurStyle(method="fill"))


def test_fractional_regions_raise_at_the_core_entry_point():
    img = _noise(12, 12, seed=7)
    with pytest.raises(ValueError, match="whole pixels"):
        blur_regions(img, [(2, 2, 0.4, 5)], BlurStyle(method="fill"))


def test_regions_outside_the_frame_are_still_skipped_silently():
    """Clipping is not malformed input: a screen-absolute region may
    legitimately miss a sub-region capture."""
    img = _noise(12, 12, seed=7)
    before = img.copy()
    blur_regions(img, [(100, 100, 5, 5)], BlurStyle(method="fill"))
    assert (img == before).all()


def test_core_entry_point_materialises_a_generator():
    img = _noise(12, 12, seed=7)
    blur_regions(img, (r for r in [(0, 0, 4, 4)]),
                 BlurStyle(method="fill", color=(1, 2, 3)))
    assert (img[0:4, 0:4, B] == 1).all()


def test_regions_are_all_validated_before_any_pixel_changes():
    """A lazy `for region in regions` would redact, then raise part-way.

    Validating up front means a bad rectangle anywhere in the list leaves
    the frame exactly as it was, rather than half-redacted.
    """
    img = _noise(12, 12, seed=7)
    before = img.copy()
    regions = (r for r in [(0, 0, 4, 4), (6, 6, 0, 2)])
    with pytest.raises(ValueError, match="positive"):
        blur_regions(img, regions, BlurStyle(method="fill", color=(1, 2, 3)))
    assert (img == before).all(), "the first region was applied before raising"


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

@pytest.mark.parametrize("kwargs", [
    {"method": "box", "radius": 0},
    {"method": "gaussian", "radius": 0},
    {"method": "pixelate", "block": 1},
])
def test_identity_settings_are_rejected_not_silently_ignored(kwargs):
    """A style that leaves pixels readable is the failure mode that leaks."""
    with pytest.raises(ValueError, match="unchanged"):
        BlurStyle(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"method": "fill", "radius": 0},
    {"method": "fill", "block": 1},
    {"method": "box", "block": 1},
    {"method": "pixelate", "radius": 0},
])
def test_values_irrelevant_to_the_method_are_left_alone(kwargs):
    """Only the settings the chosen method actually reads are validated."""
    BlurStyle(**kwargs)


def test_blur_style_cannot_be_weakened_after_construction():
    """Frozen: validating at construction is pointless if it can be undone."""
    import dataclasses

    style = BlurStyle(method="box", radius=4)
    for field, value in (("radius", 0), ("block", 1), ("method", "swirl")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(style, field, value)


class _LooseStyle:
    """A duck-typed stand-in for BlurStyle — deliberately not accepted."""

    def __init__(self, method="box", radius=12, block=16, passes=3,
                 color=(0, 0, 0)):
        self.method = method
        self.radius = radius
        self.block = block
        self.passes = passes
        self.color = color


@pytest.mark.parametrize("style", [
    "fill",                 # the method name instead of a style
    {},
    object(),
    _LooseStyle(),          # right shape, still not a BlurStyle
    BlurStyle,              # the class rather than an instance
])
def test_blur_regions_rejects_anything_that_is_not_a_blur_style(style):
    """Reading attributes off a stray object turned it into a box blur.

    ``blur_regions(img, None, "fill")`` used to fall back to every
    default and softly blur the region, when the caller had asked for the
    one method that actually destroys pixels.
    """
    img = _noise(16, 16, seed=8)
    before = img.copy()
    with pytest.raises(TypeError, match="BlurStyle"):
        blur_regions(img, None, style)
    assert (img == before).all(), "the frame was touched before rejecting"


@pytest.mark.parametrize("kwargs,match", [
    ({"method": "swirl"}, "unknown blur method"),
    ({"method": "gaussian", "passes": 0}, "passes"),
    ({"method": "fill", "color": (1, 2)}, "B, G, R"),
    ({"method": "fill", "color": 300}, "B, G, R"),
])
def test_blur_style_rejects_structurally_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BlurStyle(**kwargs)


def test_blur_style_rejects_an_image_as_a_colour():
    """A fill whose "colour" is an image assigns the frame back to itself."""
    with pytest.raises(ValueError, match="B, G, R"):
        BlurStyle(method="fill", color=numpy.zeros((8, 8, 3), numpy.uint8))


@pytest.mark.parametrize("kwargs", [
    {"radius": 1.5},
    {"method": "pixelate", "block": 2.5},
    {"method": "gaussian", "passes": 1.5},
    {"radius": float("nan")},
    {"method": "pixelate", "block": float("nan")},
    {"radius": "12"},
])
def test_blur_style_rejects_non_whole_controls(kwargs):
    """These used to construct, then fail with TypeError deep in a capture.

    NaN is the interesting one: it evades every ``<`` range check.
    """
    with pytest.raises(ValueError, match="whole number"):
        BlurStyle(**kwargs)


def test_blur_style_accepts_integral_floats():
    assert BlurStyle(radius=8.0).radius == 8


# --------------------------------------------------------------------
# Exact numerical oracles — the cumsum implementation checked against a
# brute-force reference rather than against itself
# --------------------------------------------------------------------

@pytest.mark.parametrize("shape,radius", [
    ((9, 11), 1),      # interior + head + tail all exercised
    ((9, 11), 3),
    ((8, 8), 4),       # 2*radius == n, the gather fallback on both axes
    ((8, 8), 7),       # kernel far wider than the region
    ((1, 12), 2),      # single row: fallback on y, sliced path on x
    ((12, 1), 2),      # single column: the mirror case
    ((5, 5), 2),       # 2*radius == n - 1, the tightest sliced case
])
def test_box_blur_matches_a_brute_force_reference(shape, radius):
    img = _noise(shape[0], shape[1], seed=sum(shape) + radius)
    expected = numpy.stack(
        [_reference_box(img[..., c], radius) for c in range(3)], axis=-1
    )
    blur_regions(img, None, BlurStyle(method="box", radius=radius))
    # Both round the same way; allow one unit for float32 vs float64.
    assert numpy.abs(
        img[..., :3].astype(numpy.int32) - numpy.floor(expected + 0.5)
    ).max() <= 1


def test_pixelate_tiles_hold_the_correct_mean():
    """Ragged edge tiles must average their own pixels, not a padded block."""
    img = _noise(10, 7, seed=21)
    before = img[..., :3].astype(numpy.float64).copy()
    blur_regions(img, None, BlurStyle(method="pixelate", block=4))
    for y0, y1 in ((0, 4), (4, 8), (8, 10)):
        for x0, x1 in ((0, 4), (4, 7)):
            expected = before[y0:y1, x0:x1].mean(axis=(0, 1))
            got = img[y0:y1, x0:x1, :3]
            assert (got == got[0, 0]).all(), "tile is not uniform"
            assert numpy.abs(
                got[0, 0].astype(numpy.float64) - numpy.floor(expected + 0.5)
            ).max() <= 1


def test_gaussian_variance_tracks_the_requested_radius():
    """Combined variance must stay inside a symmetric envelope.

    Not "never stronger": integer radii cannot hit the target exactly, so
    the result may land up to a quarter either side of it.
    """
    from fastgrab.effects import _pass_radii

    for radius in range(1, 33):
        for passes in (1, 2, 3, 5):
            radii = _pass_radii(radius, passes)
            assert radii, "no passes produced"
            assert all(r >= 1 for r in radii)
            assert len(radii) <= passes
            target = ((2 * radius + 1) ** 2 - 1) / 12.0
            got = sum((2 * r + 1) ** 2 - 1 for r in radii) / 12.0
            assert abs(got - target) <= 0.25 * target, (
                "radius={} passes={} -> {} (variance {} vs {})".format(
                    radius, passes, radii, got, target
                )
            )


def test_gaussian_radius_one_does_not_triple_the_strength():
    """Regression: flooring each pass at radius 1 made this 3x too strong."""
    from fastgrab.effects import _pass_radii

    assert _pass_radii(1, 3) == [1]


def test_pass_radii_compares_both_integer_neighbours():
    """Regression: rounding the ideal radius dropped a pass unnecessarily.

    At radius 19 over 24 passes the rounded radius (4) sits outside the
    tolerance envelope while its floor (3) sits inside, so rounding gave
    23 passes where 24 fit. Picking by variance rather than by proximity
    keeps them.
    """
    from fastgrab.effects import _pass_radii

    assert _pass_radii(19, 24) == [3] * 24


# --------------------------------------------------------------------
# Region clipping — non-zero origin and frame-spanning cases
# --------------------------------------------------------------------

def test_partial_overlap_with_a_nonzero_origin():
    img = numpy.full((20, 20, 4), 50, numpy.uint8)
    # Screen rect (95, 90, 10, 10) against a frame whose origin is
    # (100, 100): only the bottom-right 5x0... nothing overlaps in y.
    blur_regions(img, [(95, 90, 10, 10)],
                 BlurStyle(method="fill", color=(9, 9, 9)),
                 origin=(100, 100))
    assert (img[..., B] == 50).all(), "a rect above the frame was applied"

    # Now one that clips against the frame's top-left corner.
    blur_regions(img, [(95, 95, 10, 10)],
                 BlurStyle(method="fill", color=(9, 9, 9)),
                 origin=(100, 100))
    assert (img[0:5, 0:5, B] == 9).all()
    assert (img[5:, 5:, B] == 50).all()


def test_region_spanning_the_whole_frame_and_beyond():
    img = _noise(12, 12, seed=31)
    blur_regions(img, [(-50, -50, 500, 500)],
                 BlurStyle(method="fill", color=(3, 3, 3)))
    assert (img[..., 0:3] == 3).all()


def test_region_touching_only_the_last_pixel():
    img = numpy.full((10, 10, 4), 50, numpy.uint8)
    blur_regions(img, [(9, 9, 1, 1)], BlurStyle(method="fill", color=(1, 1, 1)))
    assert img[9, 9, B] == 1
    assert (img[0:9, :, B] == 50).all()
    assert (img[:, 0:9, B] == 50).all()


# --------------------------------------------------------------------
# Scratch reuse — that it is actually used, not merely harmless
# --------------------------------------------------------------------

def test_scratch_buffers_are_actually_reused():
    """Guards against the scratch dict being accepted and then ignored."""
    style = BlurStyle(method="box", radius=3)
    regions = [(2, 2, 16, 12)]
    scratch = {}
    blur_regions(_noise(20, 20, seed=41), regions, style, scratch=scratch)
    assert scratch, "nothing was cached"
    ids_before = {key: id(val) for key, val in scratch.items()}
    blur_regions(_noise(20, 20, seed=42), regions, style, scratch=scratch)
    assert {key: id(val) for key, val in scratch.items()} == ids_before, (
        "second call replaced the cached buffers instead of reusing them"
    )


@pytest.mark.parametrize("method", ["box", "gaussian", "pixelate"])
def test_steady_state_allocates_no_per_region_work_buffers(method):
    """Warm calls must not allocate a buffer that scales with the region.

    Not zero bytes: numpy keeps a fixed iteration buffer for ufuncs whose
    output is a strided view, and each call builds a few small view
    objects. What must not happen is a work buffer proportional to the
    region — so this measures a small and a large region and checks the
    peak stays flat, which a per-region allocation could not do.
    """
    import tracemalloc

    style = BlurStyle(method=method, radius=4, block=4)

    def warm_peak(width, height):
        img = _noise(height + 16, width + 16, seed=width)
        regions = [(8, 8, width, height)]
        scratch = {}
        for _ in range(3):                  # warm every cached buffer
            blur_regions(img, regions, style, scratch=scratch)
        keys = set(scratch)
        tracemalloc.start()
        for _ in range(5):
            blur_regions(img, regions, style, scratch=scratch)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert set(scratch) == keys, "the scratch dict kept growing"
        return peak

    small = warm_peak(128, 96)
    large = warm_peak(512, 384)            # 16x the pixels
    # One un-cached float32 plane for the large region would be 768 KiB.
    assert large < 256 * 1024, (
        "{} peaked at {:.0f} KiB on the large region".format(
            method, large / 1024.0
        )
    )
    assert large < small + 64 * 1024, (
        "{} allocation scales with the region: {:.0f} KiB -> {:.0f} KiB"
        .format(method, small / 1024.0, large / 1024.0)
    )


# --------------------------------------------------------------------
# Scratch equivalence and bounding (restored — an earlier edit in this
# branch dropped these three when rewriting an adjacent block)
# --------------------------------------------------------------------

def test_blur_regions_returns_the_same_array_object():
    img = _noise(8, 8, seed=9)
    assert blur_regions(img, None, BlurStyle(method="fill")) is img


@pytest.mark.parametrize("style", [
    BlurStyle(method="gaussian", radius=4),
    BlurStyle(method="box", radius=6),
    BlurStyle(method="pixelate", block=5),
])
def test_scratch_reuse_does_not_change_the_result(style):
    """The reused work buffers must be an optimisation, nothing more."""
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
    """A caller blurring a different size every frame must not grow it."""
    style = BlurStyle(radius=2)
    scratch = {}
    for size in range(4, 60):
        blur_regions(_noise(size, size, seed=size), None, style,
                     scratch=scratch)
    assert len(scratch) <= 32


def test_scratch_dict_stays_bounded_for_pixelate():
    style = BlurStyle(method="pixelate", block=4)
    scratch = {}
    for size in range(8, 60):
        blur_regions(_noise(size, size, seed=size), None, style,
                     scratch=scratch)
    assert len(scratch) <= 32


def test_exhausted_region_iterator_raises_instead_of_doing_nothing():
    """A generator reused across calls would silently stop redacting."""
    img = _noise(16, 16, seed=12)
    gen = (r for r in [(0, 0, 4, 4)])
    blur_regions(img, gen, BlurStyle(method="fill", color=(1, 2, 3)))
    with pytest.raises(ValueError, match="empty or already consumed"):
        blur_regions(img, gen, BlurStyle(method="fill", color=(1, 2, 3)))


# --------------------------------------------------------------------
# The randomised mosaic modes
# --------------------------------------------------------------------

RANDOM_PIXELATE = ("pixelate-random", "pixelate-random-shuffle")


@pytest.mark.parametrize("method", RANDOM_PIXELATE)
def test_random_modes_make_each_tile_uniform(method):
    img = _noise(16, 16, seed=60)
    blur_regions(img, None, BlurStyle(method=method, block=4))
    for y in range(0, 16, 4):
        for x in range(0, 16, 4):
            tile = img[y:y + 4, x:x + 4, :3]
            assert (tile == tile[0, 0]).all()


def test_pixelate_random_does_not_depend_on_the_content():
    """The strong claim: the output is a function of the seed alone.

    That is what puts it alongside fill rather than alongside pixelate —
    two completely different regions must come out identical.
    """
    style = BlurStyle(method="pixelate-random", block=4)
    a = _noise(24, 24, seed=61)
    b = _noise(24, 24, seed=62)
    assert not numpy.array_equal(a[..., :3], b[..., :3])
    blur_regions(a, None, style)
    blur_regions(b, None, style)
    assert numpy.array_equal(a[..., :3], b[..., :3])


def test_pixelate_random_shuffle_keeps_the_palette_and_drops_the_layout():
    """Real tile colours, permuted positions.

    The multiset of tile colours is preserved exactly — that is the
    documented weakness of this mode — while the arrangement is not.
    """
    plain = _noise(32, 32, seed=63)
    shuffled = plain.copy()
    blur_regions(plain, None, BlurStyle(method="pixelate", block=8))
    blur_regions(shuffled, None, BlurStyle(method="pixelate-random-shuffle",
                                           block=8))
    corner = lambda im: numpy.array(  # noqa: E731 - one tile per 8x8 block
        [[im[y, x, :3] for x in range(0, 32, 8)] for y in range(0, 32, 8)]
    ).reshape(-1, 3)
    before, after = corner(plain), corner(shuffled)
    assert not numpy.array_equal(before, after), "nothing was shuffled"
    order = lambda a: sorted(map(tuple, a))  # noqa: E731
    assert order(before) == order(after), "the tile colours changed"


@pytest.mark.parametrize("method", RANDOM_PIXELATE)
def test_random_modes_are_stable_across_frames(method):
    """Same seed, same result — a recording cannot be averaged clean."""
    style = BlurStyle(method=method, block=4)
    frames = []
    for seed in (70, 71, 72):
        img = _noise(20, 20, seed=seed)
        blur_regions(img, None, style)
        frames.append(img[..., :3].copy())
    if method == "pixelate-random":
        assert numpy.array_equal(frames[0], frames[1])
        assert numpy.array_equal(frames[1], frames[2])
    # For the shuffle the content differs per frame, but the permutation
    # must not: the same tile index has to land in the same place.
    perm_a = blur_regions(_noise(20, 20, seed=80), None, style)[..., :3].copy()
    perm_b = blur_regions(_noise(20, 20, seed=80), None, style)[..., :3]
    assert numpy.array_equal(perm_a, perm_b)


@pytest.mark.parametrize("method", RANDOM_PIXELATE)
def test_a_different_seed_gives_a_different_result(method):
    a = _noise(24, 24, seed=90)
    b = a.copy()
    blur_regions(a, None, BlurStyle(method=method, block=4, seed=1))
    blur_regions(b, None, BlurStyle(method=method, block=4, seed=2))
    assert not numpy.array_equal(a[..., :3], b[..., :3])


@pytest.mark.parametrize("method", RANDOM_PIXELATE)
def test_random_modes_respect_the_region_and_alpha(method):
    img = _noise(40, 40, seed=91)
    before = img.copy()
    blur_regions(img, [(10, 10, 20, 20)], BlurStyle(method=method, block=5))
    assert (img[0:10] == before[0:10]).all()
    assert (img[30:] == before[30:]).all()
    assert (img[..., A] == before[..., A]).all()


@pytest.mark.parametrize("method", RANDOM_PIXELATE)
def test_random_modes_reject_a_block_that_changes_nothing(method):
    with pytest.raises(ValueError, match="unchanged"):
        BlurStyle(method=method, block=1)


def test_blur_style_rejects_a_negative_seed():
    with pytest.raises(ValueError, match="seed"):
        BlurStyle(method="pixelate-random", seed=-1)


def test_blur_style_rejects_a_non_whole_seed():
    with pytest.raises(ValueError, match="whole number"):
        BlurStyle(method="pixelate-random", seed=1.5)


# --------------------------------------------------------------------
# The image cover
# --------------------------------------------------------------------

def test_image_covers_the_region_and_ignores_the_content():
    """Content-independent, like fill: two different regions come out same."""
    style = BlurStyle(method="image", image=_cover())
    a = _noise(24, 24, seed=100)
    b = _noise(24, 24, seed=101)
    blur_regions(a, None, style)
    blur_regions(b, None, style)
    assert numpy.array_equal(a[..., :3], b[..., :3])
    assert (a[..., 2] > 0).any(), "the cover was not drawn"


def test_image_is_stretched_to_the_region():
    """A 2x2 cover over a 20x20 region: four equal quadrants."""
    cover = numpy.zeros((2, 2, 3), numpy.uint8)
    cover[0, 0] = (10, 20, 30)
    cover[0, 1] = (40, 50, 60)
    cover[1, 0] = (70, 80, 90)
    cover[1, 1] = (100, 110, 120)
    img = _noise(20, 20, seed=102)
    blur_regions(img, None, BlurStyle(method="image", image=cover))
    assert tuple(img[0, 0, :3]) == (10, 20, 30)
    assert tuple(img[0, 19, :3]) == (40, 50, 60)
    assert tuple(img[19, 0, :3]) == (70, 80, 90)
    assert tuple(img[19, 19, :3]) == (100, 110, 120)


def test_image_respects_the_region_and_alpha():
    img = _noise(40, 40, seed=103)
    before = img.copy()
    blur_regions(img, [(10, 10, 20, 20)],
                 BlurStyle(method="image", image=_cover()))
    assert (img[0:10] == before[0:10]).all()
    assert (img[30:] == before[30:]).all()
    assert (img[..., A] == before[..., A]).all()


def test_image_method_requires_an_image():
    with pytest.raises(ValueError, match="needs image="):
        BlurStyle(method="image")


@pytest.mark.parametrize("bad,match", [
    (numpy.zeros((4, 4), numpy.uint8), "H, W, 3"),
    (numpy.zeros((4, 4, 2), numpy.uint8), "H, W, 3"),
    (numpy.zeros((4, 4, 3), numpy.float32), "uint8"),
    (numpy.zeros((0, 4, 3), numpy.uint8), "at least one pixel"),
])
def test_image_rejects_a_cover_it_cannot_use(bad, match):
    with pytest.raises(ValueError, match=match):
        BlurStyle(method="image", image=bad)


def test_a_four_channel_cover_keeps_only_bgr():
    cover = numpy.zeros((2, 2, 4), numpy.uint8)
    cover[..., 0:3] = (1, 2, 3)
    cover[..., 3] = 200
    style = BlurStyle(method="image", image=cover)
    assert style.image.shape == (2, 2, 3)


def test_the_cover_is_snapshotted_not_referenced():
    """Mutating the caller's array must not change later redactions."""
    cover = _cover()
    style = BlurStyle(method="image", image=cover)
    cover[...] = 0                       # caller scribbles over their copy
    img = _noise(12, 12, seed=104)
    blur_regions(img, None, style)
    assert (img[..., 2] > 0).any(), "the redaction followed the caller's array"
    assert not style.image.flags.writeable


def test_blur_style_equality_survives_an_image_field():
    """ndarray __eq__ returns an array; comparing styles must not raise."""
    a = BlurStyle(method="image", image=_cover())
    b = BlurStyle(method="image", image=_cover())
    assert a == b        # image is compare=False, so this is well defined
    assert a != BlurStyle(method="fill")


# --------------------------------------------------------------------
# How a cover image is mapped onto a region it does not match
# --------------------------------------------------------------------

def _gradient_cover(h, w):
    """A cover whose row and column are recoverable from any pixel.

    Blue encodes the source column, red the source row, so a covered
    frame says exactly which source pixel landed where — which is what
    makes the aspect-ratio assertions below possible.
    """
    img = numpy.zeros((h, w, 3), numpy.uint8)
    img[..., 0] = numpy.arange(w, dtype=numpy.uint8)[None, :]
    img[..., 2] = numpy.arange(h, dtype=numpy.uint8)[:, None]
    return img


def test_the_default_fit_does_not_distort():
    """The regression: a square cover in a wide region used to be squashed."""
    assert BlurStyle(method="image", image=_gradient_cover(8, 8)).image_fit \
        == "crop"


def test_crop_preserves_aspect_and_fills_the_region():
    """Cover: every pixel of the region is painted, nothing is stretched."""
    cover = _gradient_cover(64, 64)                # square
    img = numpy.zeros((20, 200, 4), numpy.uint8)   # very wide region
    blur_regions(img, None,
                 BlurStyle(method="image", image=cover, image_fit="crop"))
    # Full coverage: no pixel left at the original zero-with-alpha state.
    assert (img[..., 0:3] != 0).any(axis=2).all()
    # A square source in a 10:1 region keeps its scale, so the source
    # columns sampled span the full width while the rows are a thin band.
    col_span = int(img[..., 0].max()) - int(img[..., 0].min())
    row_span = int(img[..., 2].max()) - int(img[..., 2].min())
    assert col_span > 55, col_span          # whole width used
    assert row_span < 12, row_span          # only a slice of the height


def test_fit_preserves_aspect_and_pads_the_rest():
    """Contain: the whole picture is visible, margins take --blur-color."""
    cover = _gradient_cover(64, 64)
    img = numpy.zeros((20, 200, 4), numpy.uint8)
    blur_regions(img, None, BlurStyle(method="image", image=cover,
                                      image_fit="fit", color=(7, 8, 9)))
    # The padding colour appears at the far left and right.
    assert tuple(img[10, 0, :3]) == (7, 8, 9)
    assert tuple(img[10, 199, :3]) == (7, 8, 9)
    # The whole source is present in the inset, both axes.
    assert int(img[..., 2].max()) > 55, "rows were cropped"
    assert int(img[..., 0].max()) > 55, "columns were cropped"


def test_stretch_distorts_to_the_exact_shape():
    """The old behaviour, still available and still the only distorting one."""
    cover = _gradient_cover(64, 64)
    img = numpy.zeros((20, 200, 4), numpy.uint8)
    blur_regions(img, None,
                 BlurStyle(method="image", image=cover, image_fit="stretch"))
    # Both source axes are spanned across the whole region — that is the
    # distortion: 64 rows squeezed into 20, 64 columns spread over 200.
    assert int(img[..., 2].max()) > 55
    assert int(img[..., 0].max()) > 55
    assert (img[..., 0:3] != 0).any(axis=2).all()


def test_tile_repeats_at_the_source_scale():
    cover = _gradient_cover(8, 8)
    img = numpy.zeros((8, 40, 4), numpy.uint8)
    blur_regions(img, None,
                 BlurStyle(method="image", image=cover, image_fit="tile"))
    # Column 0 of the source reappears every 8 pixels, unscaled.
    for x in (0, 8, 16, 24, 32):
        assert img[0, x, 0] == 0, x
    assert img[0, 7, 0] == 7


@pytest.mark.parametrize("fit", list(IMAGE_FITS))
def test_every_fit_covers_the_whole_region(fit):
    """Whatever the mapping, nothing underneath may show through."""
    img = _noise(30, 90, seed=110)
    before = img.copy()
    blur_regions(img, [(5, 5, 60, 20)],
                 BlurStyle(method="image", image=_gradient_cover(16, 40),
                           image_fit=fit, color=(1, 1, 1)))
    covered = img[5:25, 5:65, :3]
    original = before[5:25, 5:65, :3]
    assert not numpy.array_equal(covered, original)
    # and only that rectangle moved
    assert (img[0:5] == before[0:5]).all()
    assert (img[25:] == before[25:]).all()
    assert (img[..., A] == before[..., A]).all()


@pytest.mark.parametrize("fit", list(IMAGE_FITS))
def test_a_cover_larger_than_the_region_is_handled(fit):
    img = _noise(12, 12, seed=111)
    blur_regions(img, None, BlurStyle(method="image",
                                      image=_gradient_cover(200, 200),
                                      image_fit=fit))


def test_an_unknown_fit_is_rejected():
    with pytest.raises(ValueError, match="unknown image fit"):
        BlurStyle(method="image", image=_gradient_cover(4, 4),
                  image_fit="squish")
