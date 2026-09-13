"""Smoke tests for fastgrab.recording.

Pure-numpy overlay tests, filter-assembly tests, and CLI parsing tests
run anywhere. Tests that actually spawn ffmpeg are skipped when ffmpeg
is not on PATH; recorder tests additionally need an X11 DISPLAY — i.e.
inside ``docker compose run --rm test``.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy
import pytest

from fastgrab.recording import (
    SUBTITLE_BACKENDS,
    ClickStyle,
    FfmpegEncoder,
    Recorder,
    Subtitle,
    SubtitleStyle,
    build_ass_document,
    infer_codec,
)
from fastgrab.recording import cli as recording_cli
from fastgrab.recording import clicks as click_mod
from fastgrab.recording import encoder as encoder_mod
from fastgrab.recording import recorder as recorder_mod
from fastgrab.recording import subtitles as subtitles_mod


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not available on PATH",
)


def _x11_available():
    return bool(os.environ.get("DISPLAY"))


def test_infer_codec_extensions():
    assert infer_codec("/tmp/clip.mp4") == "mp4"
    assert infer_codec("clip.WebM") == "webm"
    assert infer_codec("anim.gif") == "gif"
    with pytest.raises(ValueError):
        infer_codec("clip.mkv")


@requires_ffmpeg
def test_encoder_writes_mp4(tmp_path):
    out = tmp_path / "encoder.mp4"
    width, height, fps = 64, 48, 10
    enc = FfmpegEncoder(str(out), width, height, fps=fps)
    frame = numpy.zeros((height, width, 4), dtype=numpy.uint8)
    frame[..., 2] = 255  # red in BGRA
    with enc:
        for _ in range(8):
            enc.write_frame(frame)
    assert out.exists()
    assert out.stat().st_size > 0


def test_encoder_validates_fps_and_dimensions(tmp_path):
    out = str(tmp_path / "x.mp4")
    for bad_fps in (0, -1, 1.5, "30", True):
        with pytest.raises(ValueError, match="fps"):
            FfmpegEncoder(out, 64, 48, fps=bad_fps)
    assert FfmpegEncoder(out, 64, 48, fps=30.0).fps == 30
    # yuv420p codecs need even dimensions; fail at construction, not in
    # ffmpeg's stderr after the first frame.
    with pytest.raises(ValueError, match="even"):
        FfmpegEncoder(out, 63, 48)
    with pytest.raises(ValueError, match="even"):
        FfmpegEncoder(str(tmp_path / "x.webm"), 64, 47)
    with pytest.raises(ValueError, match="positive"):
        FfmpegEncoder(out, 0, 48)
    # gif has no such constraint.
    FfmpegEncoder(str(tmp_path / "x.gif"), 63, 47)


def test_recorder_validates_fps_before_opening_display(tmp_path):
    # Must raise even with no DISPLAY: validation happens before the
    # Screenshot backend is constructed.
    with pytest.raises(ValueError, match="fps"):
        Recorder(str(tmp_path / "x.mp4"), fps=0)


@requires_ffmpeg
def test_encoder_accepts_non_contiguous_and_rejects_wrong_dtype(tmp_path):
    out = tmp_path / "views.mp4"
    enc = FfmpegEncoder(str(out), 32, 24, fps=10)
    wide = numpy.zeros((24, 64, 4), dtype=numpy.uint8)
    strided = wide[:, ::2]          # right shape, not C-contiguous
    assert not strided.flags.c_contiguous
    with enc:
        for _ in range(4):
            enc.write_frame(strided)
        with pytest.raises(ValueError, match="uint8"):
            enc.write_frame(numpy.zeros((24, 32, 4), dtype=numpy.float32))
    assert out.stat().st_size > 0


@requires_ffmpeg
def test_encoder_rejects_wrong_shape(tmp_path):
    out = tmp_path / "shape.mp4"
    enc = FfmpegEncoder(str(out), 32, 32, fps=10)
    bad = numpy.zeros((16, 32, 4), dtype=numpy.uint8)
    with enc:
        with pytest.raises(ValueError):
            enc.write_frame(bad)


def test_drawtext_filter_assembled_when_font_present(tmp_path, monkeypatch):
    # Force a fake font path so the test is deterministic regardless
    # of host font config.
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    enc = FfmpegEncoder(
        str(tmp_path / "x.mp4"), 64, 48, fps=10,
        title="Hello: world", overlay_text="watermark",
    )
    argv = enc._build_argv()
    # The two drawtext filters end up combined in a single -vf arg.
    assert "-vf" in argv
    vf_value = argv[argv.index("-vf") + 1]
    assert "drawtext=" in vf_value
    # title text gets escaped — the colon must not appear unescaped.
    assert "Hello\\: world" in vf_value
    assert "watermark" in vf_value
    # First-3-seconds gating uses lt(t,N).
    assert "lt(t," in vf_value


def test_drawtext_skipped_when_no_font(tmp_path, monkeypatch):
    monkeypatch.setattr(encoder_mod, "_FONT_CANDIDATES", ())
    monkeypatch.delenv("FASTGRAB_FONT", raising=False)
    enc = FfmpegEncoder(
        str(tmp_path / "x.mp4"), 64, 48, fps=10, title="Hello",
    )
    argv = enc._build_argv()
    assert "-vf" not in argv  # no font → no drawtext filter


def test_overlay_clicks_draws_bright_pixels():
    frame = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
    events = [{"x": 50, "y": 50, "t_press": time.monotonic() - 0.05}]
    click_mod.overlay_clicks(frame, events, bbox_origin=(0, 0))
    # Some pixels in the centre region should now carry the BGR ring colour.
    centre = frame[40:60, 40:60]
    assert centre.max() > 0, "expected the click ring to colour pixels"


def test_overlay_clicks_respects_bbox_origin():
    frame = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
    # A click at screen (200, 200) inside a region whose top-left is
    # (180, 180) should land at frame coords (20, 20).
    events = [{"x": 200, "y": 200, "t_press": time.monotonic() - 0.05}]
    click_mod.overlay_clicks(frame, events, bbox_origin=(180, 180))
    # The ring is drawn around (20, 20) — pixels far from there should
    # remain untouched.
    far = frame[40:50, 40:50]
    assert far.max() == 0
    near = frame[10:30, 10:30]
    assert near.max() > 0


def test_overlay_clicks_drops_expired():
    frame = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
    # An event older than CLICK_LIFETIME should not draw anything.
    events = [{
        "x": 25, "y": 25,
        "t_press": time.monotonic() - (click_mod.CLICK_LIFETIME + 0.1),
    }]
    click_mod.overlay_clicks(frame, events)
    assert frame.max() == 0


@pytest.mark.parametrize("pattern", list(click_mod.CLICK_PATTERNS))
def test_click_patterns_paint_near_event(pattern):
    frame = numpy.zeros((200, 200, 4), dtype=numpy.uint8)
    events = [{"x": 100, "y": 100, "t_press": time.monotonic() - 0.1}]
    style = ClickStyle(pattern=pattern)
    click_mod.overlay_clicks(frame, events, style=style)
    near = frame[60:140, 60:140]
    assert near.max() > 0, "pattern {!r} painted nothing".format(pattern)
    # The animation is bounded by radius1 (~60 px) — corners stay black.
    assert frame[0:20, 0:20].max() == 0
    assert frame[180:200, 180:200].max() == 0


def test_click_concentric_uses_multiple_radii():
    # Mid-animation, the three phase-offset rings should paint pixels
    # at distinctly different distances from the click point.
    frame = numpy.zeros((300, 300, 4), dtype=numpy.uint8)
    style = ClickStyle(pattern="concentric")
    events = [{"x": 150, "y": 150, "t_press": time.monotonic() - 0.2}]
    click_mod.overlay_clicks(frame, events, style=style)
    yy, xx = numpy.nonzero(frame[..., 0])
    dists = numpy.sqrt((xx - 150) ** 2 + (yy - 150) ** 2)
    assert dists.max() - dists.min() > 20


def test_click_style_color_honored():
    frame = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
    # Pure red in BGR: B=0, G=0, R=255.
    style = ClickStyle(pattern="circle", color=(0, 0, 255))
    events = [{"x": 50, "y": 50, "t_press": time.monotonic() - 0.05}]
    click_mod.overlay_clicks(frame, events, style=style)
    centre = frame[45:55, 45:55]
    assert centre[..., 2].max() > 200  # red channel painted
    assert centre[..., 0].max() == 0   # blue untouched


def test_click_style_rejects_unknown_pattern():
    with pytest.raises(ValueError):
        ClickStyle(pattern="spiral")


def test_draw_cursor_stamps_arrow():
    frame = numpy.zeros((100, 100, 4), dtype=numpy.uint8)
    click_mod.draw_cursor(frame, 40, 40)
    # Arrow occupies a small box below-right of the tip.
    region = frame[40:62, 40:55]
    assert region.max() > 0
    # White fill → all BGR channels painted somewhere.
    assert frame[..., 0].max() == 255
    # Far corner untouched.
    assert frame[90:100, 90:100].max() == 0


def test_draw_cursor_respects_bbox_origin():
    frame = numpy.zeros((50, 50, 4), dtype=numpy.uint8)
    click_mod.draw_cursor(frame, 200, 200, bbox_origin=(180, 180))
    assert frame[20:40, 20:35].max() > 0
    assert frame[0:10, 0:10].max() == 0


def test_draw_cursor_clips_at_edges():
    frame = numpy.zeros((30, 30, 4), dtype=numpy.uint8)
    # Tip at the very corner and fully outside — neither may raise.
    click_mod.draw_cursor(frame, 0, 0)
    click_mod.draw_cursor(frame, -50, -50)
    click_mod.draw_cursor(frame, 1000, 1000)


def test_subtitle_requires_end_after_start():
    with pytest.raises(ValueError):
        Subtitle(text="nope", start=2.0, end=2.0)


def test_build_subtitle_filters(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    subs = [Subtitle(text="Hello: world", start=1.5, end=4.0)]
    vf = subtitles_mod.build_subtitle_filters(subs)
    assert "drawtext=" in vf
    assert "between(t,1.5,4.0)" in vf
    # Colon in the text must be escaped for the filter parser.
    assert "Hello\\: world" in vf
    # The path is escaped like the text — on Windows it contains ':' and
    # '\\', so compare against the escaped form, not the raw string.
    assert "fontfile=" + encoder_mod._escape_filter_path(str(fake_font)) in vf


def test_build_subtitle_filters_style_overrides(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    style = SubtitleStyle(
        font_size=40, font_color="yellow", box_color="blue@0.5",
        position="top",
    )
    subs = [Subtitle(text="styled", start=0.0, end=1.0)]
    vf = subtitles_mod.build_subtitle_filters(subs, style)
    assert "fontcolor=yellow" in vf
    assert "fontsize=40" in vf
    assert "boxcolor=blue@0.5" in vf
    assert "y=30" in vf  # top placement


def test_build_subtitle_filters_empty_and_fontless(monkeypatch):
    assert subtitles_mod.build_subtitle_filters([]) is None
    monkeypatch.setattr(encoder_mod, "_FONT_CANDIDATES", ())
    monkeypatch.delenv("FASTGRAB_FONT", raising=False)
    subs = [Subtitle(text="x", start=0.0, end=1.0)]
    assert subtitles_mod.build_subtitle_filters(subs) is None


def test_font_path_is_escaped_as_a_filter_value(tmp_path, monkeypatch):
    """A font path is an option value, not drawtext text.

    Escaped the old way -- the same single pass used for text -- a path
    containing an apostrophe, colon, comma or bracket made ffmpeg reject
    the whole filtergraph, and one containing a backslash rendered
    different pixels. Escaping for both of ffmpeg's parse passes fixes
    all five; see test_a_font_path_with_awkward_characters_still_renders.
    """
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    vf = encoder_mod._build_drawtext_filter(title="hi")
    assert "fontfile=" + encoder_mod._escape_filter_path(str(fake_font)) in vf


def test_encoder_argv_includes_subtitles(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    enc = FfmpegEncoder(
        str(tmp_path / "x.mp4"), 64, 48, fps=10,
        subtitles=[Subtitle(text="sub", start=0.5, end=2.0)],
    )
    argv = enc._build_argv()
    vf_value = argv[argv.index("-vf") + 1]
    assert "between(t,0.5,2.0)" in vf_value
    assert "sub" in vf_value


# --------------------------------------------------------------------------
# ASS subtitle backend
# --------------------------------------------------------------------------


def _ass_lines(document, prefix):
    """Every line of ``document`` starting with ``prefix``, prefix stripped."""
    return [
        line[len(prefix):] for line in document.splitlines()
        if line.startswith(prefix)
    ]


def _ass_style_row(document):
    """The ``Style:`` line as a ``{field name: value}`` mapping."""
    names = [name.strip() for name in _ass_lines(document, "Format: ")[0].split(",")]
    values = _ass_lines(document, "Style: ")[0].split(",")
    assert len(names) == len(values), "Style: row does not match its Format:"
    return dict(zip(names, values))


def test_ass_document_structure(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    subs = [
        Subtitle(text="first", start=0.0, end=1.5),
        Subtitle(text="second", start=1.5, end=3.0),
    ]
    doc = build_ass_document(subs, width=640, height=480)
    # The three sections an ASS script needs, in the order players expect.
    assert doc.index("[Script Info]") < doc.index("[V4+ Styles]")
    assert doc.index("[V4+ Styles]") < doc.index("[Events]")
    assert "ScriptType: v4.00+" in doc
    # PlayRes has to match the video, or libass rescales every size and
    # margin we computed in pixels.
    assert "PlayResX: 640" in doc
    assert "PlayResY: 480" in doc
    # Style:/Dialogue: rows are positional, so their field counts must
    # match the Format: lines that declare them.
    _ass_style_row(doc)  # asserts the style row lines up
    events_format = _ass_lines(doc, "Format: ")[1].split(",")
    dialogues = _ass_lines(doc, "Dialogue: ")
    assert len(dialogues) == 2
    for line in dialogues:
        # Text is the last field and may itself contain commas.
        assert len(line.split(",", len(events_format) - 1)) == len(events_format)
    assert dialogues[0].startswith("0,0:00:00.00,0:00:01.50,Default,,0,0,0,,first")
    assert doc.endswith("\n")


@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00:00.00"),
    (0.25, "0:00:00.25"),
    (1.5, "0:00:01.50"),
    (59.999, "0:01:00.00"),   # rounds up into the next minute
    (3600.0, "1:00:00.00"),
    (3661.5, "1:01:01.50"),
    (36000, "10:00:00.00"),   # hours are not capped at a single digit
    (-2.0, "0:00:00.00"),     # Subtitle allows a negative start; clamp it
])
def test_ass_timecode_boundaries(seconds, expected):
    assert subtitles_mod._ass_time(seconds) == expected


@pytest.mark.parametrize("spec,expected", [
    ("white", "&H00FFFFFF"),
    ("black", "&H00000000"),
    # ASS alpha is transparency, so 0.55 opaque becomes 0x73, not 0x8C.
    ("black@0.55", "&H73000000"),
    # &HAABBGGRR is byte-reversed: pure red lands in the last byte.
    ("red", "&H000000FF"),
    ("yellow", "&H0000FFFF"),
    ("0xFF8000", "&H000080FF"),
    ("#00FF00", "&H0000FF00"),
    ("0x00FF0080", "&H7F00FF00"),   # trailing AA is opacity
    ("red@0x80", "&H7F0000FF"),     # hex alpha suffix
    ("white@0", "&HFFFFFFFF"),      # fully transparent
    ("white@2", "&H00FFFFFF"),      # out-of-range opacity clamps
])
def test_ass_color_conversion(spec, expected):
    assert subtitles_mod._ass_color(spec) == expected


def test_ass_color_rejects_what_it_cannot_translate():
    # drawtext passes colour names straight to ffmpeg, so it accepts all
    # ~150 of them; the ASS backend has to convert and only knows a
    # common subset. Failing beats silently painting the wrong colour.
    with pytest.raises(ValueError, match="0xRRGGBB"):
        subtitles_mod._ass_color("chartreuse")
    with pytest.raises(ValueError, match="hexadecimal"):
        subtitles_mod._ass_color("0xZZZZZZ")
    with pytest.raises(ValueError, match="alpha"):
        subtitles_mod._ass_color("white@opaque")


def test_ass_text_escaping():
    esc = subtitles_mod._escape_ass
    # Braces open an override block and would swallow the rest of the line.
    assert esc("a {b} c") == "a \\{b\\} c"
    # Newlines cannot appear in a Dialogue line; \r\n collapses to one break.
    assert esc("one\ntwo") == "one\\Ntwo"
    assert esc("one\r\ntwo") == "one\\Ntwo"
    assert esc("one\rtwo") == "one\\Ntwo"
    # libass renders an unrecognised \x as a literal backslash, so a path
    # survives untouched — doubling it would paint two backslashes.
    assert esc("C:\\Users\\dir") == "C:\\Users\\dir"
    # ... except before n/N/h, where the pair would be eaten as a space,
    # a line break or a non-breaking space.
    # Separated by a zero-width space, not doubled: doubling leaves
    # the control sequence active and libass still broke
    # "left\\Nright" across two lines.
    assert esc("C:\\new") == "C:\\" + subtitles_mod._ZWSP + "new"
    assert esc("a\\Nb") == "a\\" + subtitles_mod._ZWSP + "Nb"
    assert esc("a\\hb") == "a\\" + subtitles_mod._ZWSP + "hb"
    # A user backslash before a brace still round-trips: libass reads the
    # '\\' as a literal backslash and then '\{' as a literal brace.
    assert esc("a\\{b") == "a\\\\{b"
    # Nothing to do for ordinary text.
    assert esc("plain text, with a comma") == "plain text, with a comma"


def test_ass_style_maps_subtitle_style():
    style = SubtitleStyle(
        font_size=40, font_color="yellow", box_color="blue@0.5",
        border=12, position="top", font_name="DejaVu Sans",
    )
    row = _ass_style_row(build_ass_document([Subtitle("x", 0.0, 1.0)], style))
    assert row["Fontname"] == "DejaVu Sans"
    assert row["Fontsize"] == "40"
    assert row["PrimaryColour"] == "&H0000FFFF"
    # BorderStyle 3 paints an opaque box filled with OutlineColour, which
    # is the closest thing ASS has to drawtext's box=1 + boxcolor.
    assert row["BorderStyle"] == "3"
    assert row["OutlineColour"] == subtitles_mod._ass_color("blue@0.5")
    assert row["Outline"] == "12"     # boxborderw becomes the box padding
    assert row["Shadow"] == "0"       # SubtitleStyle has no shadow field
    assert row["Alignment"] == "8"    # top centre
    assert row["MarginV"] == "30"     # matches drawtext's y=30

    bottom = _ass_style_row(build_ass_document([Subtitle("x", 0.0, 1.0)]))
    assert bottom["Alignment"] == "2"
    assert bottom["MarginV"] == "40"  # matches drawtext's y=h-text_h-40


def test_ass_font_name_guessed_from_font_path(tmp_path, monkeypatch):
    fake_font = tmp_path / "MyFont-Bold.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    # libass resolves families, not paths, so the stem is the best guess
    # and the containing directory is offered to it as fontsdir.
    row = _ass_style_row(build_ass_document([Subtitle("x", 0.0, 1.0)]))
    assert row["Fontname"] == "MyFont-Bold"
    assert subtitles_mod.ass_fonts_dir() == str(tmp_path)
    # A comma would shift every field after Fontname by one.
    with pytest.raises(ValueError, match="comma"):
        build_ass_document([Subtitle("x", 0.0, 1.0)],
                           SubtitleStyle(font_name="Bad, Font"))


def test_ass_document_renders_without_a_font_file(monkeypatch):
    monkeypatch.setattr(encoder_mod, "_FONT_CANDIDATES", ())
    monkeypatch.delenv("FASTGRAB_FONT", raising=False)
    subs = [Subtitle(text="x", start=0.0, end=1.0)]
    # drawtext needs a font file and skips when there is none; libass
    # falls back to a default face, so the ASS backend still renders.
    assert subtitles_mod.build_subtitle_filters(subs) is None
    assert "Style: Default,Sans," in build_ass_document(subs)
    assert subtitles_mod.ass_fonts_dir() is None


def test_ass_filter_escapes_the_path_twice():
    # ffmpeg unescapes a filter option value twice: once splitting the
    # graph into filters, once splitting a filter's args into key=value.
    # Escaping only once is the trap — ffmpeg then reads the surviving
    # quote as an opening quote and opens a different file.
    path = "/tmp/it's, a:path.ass"
    once = encoder_mod._escape_filter_value(path)
    twice = encoder_mod._escape_filter_value(once)
    vf = subtitles_mod.build_ass_filter(path)
    assert vf == "ass=filename=" + twice
    assert "\\\\\\'" in vf   # ' -> \' -> \\\'
    assert "\\\\\\:" in vf
    assert "\\\\\\," in vf
    assert path not in vf    # nothing left raw for the parser to trip on
    # Windows separators are escape characters to the filter parser too.
    assert subtitles_mod.build_ass_filter("C:\\subs\\x.ass") == (
        "ass=filename=C\\\\\\:\\\\\\\\subs\\\\\\\\x.ass"
    )
    with_dir = subtitles_mod.build_ass_filter("/a/x.ass", fontsdir="/f:onts")
    assert with_dir.endswith(":fontsdir=/f\\\\\\:onts")


def test_encoder_ass_backend_writes_and_cleans_up_the_script(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    enc = FfmpegEncoder(
        str(tmp_path / "x.mp4"), 64, 48, fps=10,
        subtitles=[Subtitle(text="sub", start=0.5, end=2.0)],
        subtitle_backend="ass",
    )
    # start() writes the script; this is that step on its own.
    script = enc._write_ass(enc.ass_document())
    argv = enc._build_argv(script)
    vf_value = argv[argv.index("-vf") + 1]
    assert vf_value.startswith("ass=filename=")
    assert "drawtext=" not in vf_value          # ASS replaces the chain

    assert os.path.exists(script)
    with open(script, encoding="utf-8") as fobj:
        doc = fobj.read()
    assert "Dialogue: 0,0:00:00.50,0:00:02.00,Default,,0,0,0,,sub" in doc
    assert "PlayResX: 64" in doc                # the encoder's frame size
    # close() without a start() still removes the temporary script.
    enc.close()
    assert not os.path.exists(script)
    assert enc._ass_path is None


def test_encoder_ass_backend_keeps_title_drawtext(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    enc = FfmpegEncoder(
        str(tmp_path / "x.mp4"), 64, 48, fps=10, title="Demo",
        subtitles=[Subtitle(text="sub", start=0.0, end=1.0)],
        subtitle_backend="ass",
    )
    argv = enc._build_argv(enc._write_ass(enc.ass_document()))
    vf_value = argv[argv.index("-vf") + 1]
    # title/overlay are a separate feature and stay on drawtext; the ASS
    # filter is appended after them in the same chain.
    assert vf_value.index("drawtext=") < vf_value.index("ass=filename=")
    enc.close()


def test_encoder_default_subtitle_backend_is_unchanged_drawtext(tmp_path, monkeypatch):
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    subs = [Subtitle(text="sub", start=0.5, end=2.0)]
    enc = FfmpegEncoder(str(tmp_path / "x.mp4"), 64, 48, fps=10, subtitles=subs)
    assert enc.subtitle_backend == "drawtext"
    argv = enc._build_argv()
    vf_value = argv[argv.index("-vf") + 1]
    # Byte-for-byte what the drawtext builder produces, and no ASS
    # anywhere: adding the backend must not perturb the default path.
    assert vf_value == subtitles_mod.build_subtitle_filters(subs)
    assert "ass=" not in vf_value
    assert enc._ass_path is None       # nothing written to disk either


def test_unknown_subtitle_backend_is_rejected(tmp_path):
    assert SUBTITLE_BACKENDS == ("drawtext", "ass")
    with pytest.raises(ValueError, match="subtitle_backend"):
        FfmpegEncoder(str(tmp_path / "x.mp4"), 64, 48, subtitle_backend="srt")
    # Recorder validates before it opens the display, like it does fps.
    with pytest.raises(ValueError, match="subtitle_backend"):
        Recorder(str(tmp_path / "x.mp4"), subtitle_backend="srt")


@requires_ffmpeg
def test_subtitle_sidecar_is_published_only_after_a_successful_encode(
    tmp_path, monkeypatch
):
    """The blocker, in one test.

    Building the command line used to write the sidecar. That is why a
    mistyped --subtitle-sidecar destroyed the named file before any
    capture happened, on the default drawtext backend. Now argv is pure
    and the sidecar appears only once ffmpeg has exited cleanly.
    """
    fake_font = tmp_path / "fake.ttf"
    fake_font.write_bytes(b"")
    monkeypatch.setenv("FASTGRAB_FONT", str(fake_font))
    sidecar = tmp_path / "clip.ass"
    subs = [Subtitle(text="sub", start=0.0, end=1.0)]
    enc = FfmpegEncoder(
        str(tmp_path / "clip.mp4"), 64, 48, fps=10, subtitles=subs,
        subtitle_sidecar=str(sidecar),
    )
    argv = enc._build_argv()
    vf_value = argv[argv.index("-vf") + 1]
    # A sidecar is orthogonal to the backend: the burn-in is still drawtext.
    assert "drawtext=" in vf_value
    assert "ass=filename=" not in vf_value
    assert not sidecar.exists(), "building argv wrote the caller's file"

    enc._build_argv = lambda *a, **k: [sys.executable, "-c",
                                       "import sys; sys.stdin.buffer.read()"]
    enc.start()
    enc.close()
    assert sidecar.exists()
    assert "[Events]" in sidecar.read_text(encoding="utf-8")
    assert enc._ass_path is None


@requires_ffmpeg
def test_subtitle_sidecar_that_cannot_be_written_reports_clearly(tmp_path):
    enc = FfmpegEncoder(
        str(tmp_path / "clip.mp4"), 64, 48, fps=10,
        subtitles=[Subtitle(text="sub", start=0.0, end=1.0)],
        subtitle_sidecar=str(tmp_path / "no-such-dir" / "clip.ass"),
    )
    enc._build_argv = lambda *a, **k: [sys.executable, "-c",
                                       "import sys; sys.stdin.buffer.read()"]
    enc.start()
    # The CLI only prints RuntimeError/ValueError, so a bad path handed
    # in by the user must not surface as a bare OSError traceback.
    with pytest.raises(RuntimeError, match="subtitle sidecar"):
        enc.close()


def test_cli_subtitle_backend_and_sidecar_flags():
    parser = recording_cli.build_parser()
    args = parser.parse_args(["--fullscreen", "-o", "x.mp4"])
    assert args.subtitle_backend == "drawtext"
    assert args.subtitle_sidecar is None
    args = parser.parse_args([
        "--fullscreen", "-o", "x.mp4", "--subtitle-backend", "ass",
        "--subtitle-font-name", "DejaVu Sans", "--subtitle-sidecar",
    ])
    assert args.subtitle_backend == "ass"
    assert args.subtitle_font_name == "DejaVu Sans"
    assert args.subtitle_sidecar == ""     # bare flag -> derive from output
    with pytest.raises(SystemExit):
        parser.parse_args(["--fullscreen", "-o", "x.mp4",
                           "--subtitle-backend", "srt"])


def test_cli_resolve_sidecar():
    assert recording_cli._resolve_sidecar(None, "demo.mp4") is None
    assert recording_cli._resolve_sidecar("", "demo.mp4") == "demo.ass"
    assert recording_cli._resolve_sidecar("", "/tmp/a.b/demo.webm") == "/tmp/a.b/demo.ass"
    assert recording_cli._resolve_sidecar("/tmp/other.ass", "demo.mp4") == "/tmp/other.ass"


@requires_ffmpeg
def test_encoder_burns_ass_subtitles(tmp_path):
    out = tmp_path / "ass.mp4"
    # Text with a brace and a backslash goes through the real libass
    # parser, which rejects a malformed script outright.
    subs = [Subtitle(text="burned {in} C:\\path", start=0.0, end=1.0)]
    enc = FfmpegEncoder(
        str(out), 64, 48, fps=10, subtitles=subs, subtitle_backend="ass",
    )
    frame = numpy.zeros((48, 64, 4), dtype=numpy.uint8)
    script = None
    with enc:
        script = enc._ass_path
        for _ in range(8):
            enc.write_frame(frame)
    assert out.exists()
    assert out.stat().st_size > 0
    assert not os.path.exists(script)   # temp script removed after encoding


@requires_ffmpeg
@pytest.mark.skipif(
    os.name == "nt", reason="':' is not a legal filename character on Windows",
)
def test_encoder_burns_ass_from_a_path_full_of_filter_metacharacters(tmp_path):
    # ':' separates filter options, ',' separates filters and "'" quotes,
    # so an under-escaped path makes ffmpeg either fail to parse the graph
    # or quietly open a file that does not exist. Both surface here as a
    # RuntimeError out of close().
    sidecar = tmp_path / "we'ird, na:me.ass"
    out = tmp_path / "nasty.mp4"
    enc = FfmpegEncoder(
        str(out), 64, 48, fps=10,
        subtitles=[Subtitle(text="hi", start=0.0, end=1.0)],
        subtitle_backend="ass", subtitle_sidecar=str(sidecar),
    )
    frame = numpy.zeros((48, 64, 4), dtype=numpy.uint8)
    with enc:
        for _ in range(6):
            enc.write_frame(frame)
    assert sidecar.exists()
    assert out.stat().st_size > 0


def test_cli_parse_subtitle():
    sub = recording_cli._parse_subtitle("1.5-4.0:Hello world")
    assert isinstance(sub, Subtitle)
    assert (sub.start, sub.end, sub.text) == (1.5, 4.0, "Hello world")
    with pytest.raises(argparse.ArgumentTypeError):
        recording_cli._parse_subtitle("not-a-subtitle")
    with pytest.raises(argparse.ArgumentTypeError):
        recording_cli._parse_subtitle("3-1:backwards")


def test_cli_parse_bgr():
    assert recording_cli._parse_bgr("255,200,0") == (255, 200, 0)
    with pytest.raises(argparse.ArgumentTypeError):
        recording_cli._parse_bgr("1,2")
    with pytest.raises(argparse.ArgumentTypeError):
        recording_cli._parse_bgr("300,0,0")


def test_cli_parse_region():
    assert recording_cli._parse_region("10,20,300,200") == (10, 20, 300, 200)
    for bad in ("1,2,3", "a,b,c,d", "-1,0,10,10", "0,-5,10,10",
                "0,0,1,10", "0,0,10,1", "0,0,0,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            recording_cli._parse_region(bad)


def test_cli_fps_must_be_positive_int():
    assert recording_cli._positive_int("30") == 30
    for bad in ("0", "-5", "abc", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            recording_cli._positive_int(bad)
    parser = recording_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--fullscreen", "-o", "x.mp4", "--fps", "0"])


class _StubRoot:
    """Stand-in for an Xlib root window: a fixed pointer, no buttons."""

    class _Pointer:
        root_x, root_y, mask = 5, 7, 0

    def query_pointer(self):
        return self._Pointer()


def test_mouse_tracker_honours_lifetime():
    now = time.monotonic()
    stale = {"x": 1, "y": 1, "t_press": now - 0.8}

    default = click_mod.MouseTracker()
    default._root = _StubRoot()
    default._events = [dict(stale)]
    assert default.poll() == []  # 0.8 s > default 0.5 s lifetime → pruned

    longer = click_mod.MouseTracker(lifetime=2.0)
    longer._root = _StubRoot()
    longer._events = [dict(stale)]
    assert len(longer.poll()) == 1  # still within the 2 s window
    assert longer.position == (5, 7)


def test_print_xbindkeys_outputs_snippet(capsys):
    rc = recording_cli.main(["--print-xbindkeys"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "xbindkeysrc" in out
    assert "fastgrab-record --gui" in out


def test_cli_requires_capture_target(capsys):
    # No --fullscreen / --region / --gui → argparse.error → SystemExit(2).
    with pytest.raises(SystemExit) as exc:
        recording_cli.main(["-o", "/tmp/whatever.mp4"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--fullscreen" in err
    assert "--region" in err


@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_bbox_rounds_odd_dims_to_even():
    rec = Recorder(
        output_path="/tmp/unused.mp4",
        bbox=(0, 0, 1089, 615),
        fps=10,
        backend="x11",
    )
    x, y, w, h = rec._resolved_bbox()
    assert (x, y) == (0, 0)
    assert (w, h) == (1088, 614)


@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_rejects_region_too_small_after_alignment(tmp_path):
    # 1x1 rounds down to 0x0 for yuv420p; must fail before ffmpeg starts.
    rec = Recorder(
        output_path=str(tmp_path / "tiny.mp4"), bbox=(0, 0, 1, 1), backend="x11",
    )
    with pytest.raises(ValueError, match="too small"):
        rec.record(duration=0.1)
    assert not (tmp_path / "tiny.mp4").exists()


@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_countdown_cancelled_records_nothing(tmp_path):
    out = tmp_path / "never.mp4"
    rec = Recorder(
        output_path=str(out), bbox=(0, 0, 120, 90), fps=10, backend="x11",
    )
    stop = threading.Event()
    stop.set()  # fire before the countdown even ticks
    stats = rec.record(countdown=5, stop_event=stop)
    # Aborted during the countdown → ffmpeg never ran, nothing on disk.
    assert stats["frames"] == 0
    assert stats["elapsed_seconds"] == 0.0
    assert not out.exists()


@requires_ffmpeg
@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorded_gif_is_not_transparent(tmp_path):
    """Every recorded GIF used to come out completely invisible.

    A captured frame's fourth byte is unused padding, not transparency —
    X11's XGetImage leaves it zero on a 24-bit visual. The encoder
    described its rawvideo input to ffmpeg as ``bgra``, so ffmpeg read
    that padding as "fully transparent". mp4 and webm never noticed
    because they force yuv420p and drop alpha, but GIF keeps it:
    ``paletteuse`` treats alpha below its default threshold of 128 as
    transparent, so 100% of the pixels in the output were.

    Asserted on the decoded image rather than on ffmpeg's argv, so it
    stays true whatever the filter chain becomes.
    """
    out = tmp_path / "clip.gif"
    rec = Recorder(
        output_path=str(out), bbox=(0, 0, 320, 240), fps=10, backend="x11",
    )
    stop = threading.Event()

    def progress(n, _elapsed):
        if n >= 3:
            stop.set()

    rec.record(duration=30.0, stop_event=stop, on_progress=progress)
    assert out.exists() and out.stat().st_size > 0

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out),
         "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        capture_output=True, check=True,
    ).stdout
    alpha = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(-1, 4)[:, 3]
    assert alpha.size > 0, "decoded no pixels from the gif"
    assert alpha.min() == 255, (
        "{:.1f}% of the gif's pixels are transparent".format(
            (alpha == 0).mean() * 100)
    )


@requires_ffmpeg
@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_smoke_mp4(tmp_path):
    out = tmp_path / "smoke.mp4"
    rec = Recorder(
        output_path=str(out),
        bbox=(0, 0, 240, 180),
        fps=10,
        backend="x11",
    )
    # Stop on a frame count, not on a clock. This used to record for a
    # fixed 0.6 s and assert >= 4 frames, which at fps=10 demands the host
    # sustain ~7 fps of capture-plus-encode — a statement about how fast
    # the machine is rather than about the recorder, and one a loaded CI
    # runner does not honour (issue #50; seen failing with `assert 1 >= 4`).
    # record() checks stop_event at the top of every iteration and calls
    # on_progress after each captured frame, so asking for the frames we
    # want is exact. `duration` stays only as a generous backstop, so a
    # genuinely broken capture loop fails the suite instead of hanging it.
    wanted = 4
    stop = threading.Event()
    seen = []

    def progress(n, _elapsed):
        seen.append(n)
        if n >= wanted:
            stop.set()

    stats = rec.record(duration=30.0, stop_event=stop, on_progress=progress)
    # A slow host now takes longer rather than failing; only a host that
    # cannot manage 4 frames in 30 s trips this, which is a real problem.
    assert stats["frames"] >= wanted
    # ffmpeg receives at least one frame per captured frame; any extras
    # are duplicates written to hold the target rate.
    assert stats["written_frames"] >= stats["frames"]
    assert stats["achieved_fps"] == pytest.approx(
        stats["frames"] / stats["elapsed_seconds"]
    )
    # on_progress fires once per captured frame with a monotonic count that
    # ends on the final total.
    assert seen == sorted(seen)
    assert seen[-1] == stats["frames"]
    assert out.exists()
    assert out.stat().st_size > 0
    if shutil.which("ffprobe"):
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(out),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "h264" in result.stdout.strip().lower()


# -------- ffmpeg's stderr must never be able to block it --------
#
# stderr used to be a subprocess.PIPE that nothing read until close().
# A pipe holds 64 KiB (F_GETPIPE_SZ on Linux); an ffmpeg that filled it
# would block writing its own diagnostics, and an ffmpeg blocked on
# stderr stops reading stdin, which blocks write_frame(). Neither side
# has a timeout, so that is a hang, not an error.
#
# -loglevel error keeps real ffmpeg quiet — mp4, webm and gif encodes of
# 120 frames each produced 0 bytes of stderr, and so did titles carrying
# glyphs the font lacks — so this was latent rather than reachable. It is
# still worth removing: the safety rested entirely on a log level nothing
# enforced, and what it guarded against was an unkillable hang. These
# tests stand in for an ffmpeg that does talk.

_NOISY_CHILD = (
    "import sys\n"
    "sys.stderr.buffer.write(b'e' * {volume})\n"
    "sys.stderr.buffer.write(b'\\nLAST-LINE-OF-STDERR\\n')\n"
    "sys.stderr.buffer.flush()\n"
    "read = 0\n"
    "while True:\n"
    "    chunk = sys.stdin.buffer.read(65536)\n"
    "    if not chunk:\n"
    "        break\n"
    "    read += len(chunk)\n"
    "sys.exit({status})\n"
)


def _encoder_over(tmp_path, volume, status=0):
    """An encoder whose 'ffmpeg' is a child with a known stderr volume."""
    enc = FfmpegEncoder(str(tmp_path / "out.mp4"), 64, 48, fps=30)
    enc._build_argv = lambda *a, **k: [
        sys.executable, "-c",
        _NOISY_CHILD.format(volume=volume, status=status),
    ]
    return enc


def _feed(enc, frames=40, timeout=60.0):
    """Write frames from a thread; return True if the writer finished."""
    frame = numpy.zeros((48, 64, 4), numpy.uint8)
    done = threading.Event()

    def run():
        try:
            for _ in range(frames):
                enc.write_frame(frame)
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    return done.wait(timeout)


@requires_ffmpeg
@pytest.mark.parametrize("volume", [16 * 1024, 512 * 1024])
def test_a_talkative_encoder_does_not_block_the_writer(tmp_path, volume):
    """The deadlock itself.

    16 KiB fits the pipe and always worked; 512 KiB does not, and used to
    wedge write_frame() forever. 40 frames of 64x48 is ~480 KiB, well
    past the 64 KiB stdin pipe, so a child that has stopped reading
    cannot be masked by buffering.
    """
    enc = _encoder_over(tmp_path, volume)
    enc.start()
    try:
        assert _feed(enc), (
            "write_frame() never returned: the encoder's stderr filled and "
            "nothing was reading it"
        )
    finally:
        enc.close()


@requires_ffmpeg
def test_a_large_stderr_is_reported_as_its_tail(tmp_path):
    """A megabyte of diagnostics must not become a megabyte of exception."""
    enc = _encoder_over(tmp_path, 512 * 1024, status=1)
    enc.start()
    assert _feed(enc)
    with pytest.raises(RuntimeError) as excinfo:
        enc.close()
    message = str(excinfo.value)
    assert "exited with status 1" in message
    # the end of the log survives — that is where ffmpeg says why it died
    assert "LAST-LINE-OF-STDERR" in message
    # and the truncation is declared rather than silent
    assert "earlier bytes omitted" in message
    assert len(message) < 128 * 1024, "the whole log went into the message"


@requires_ffmpeg
def test_a_short_stderr_is_reported_whole(tmp_path):
    """No truncation notice when nothing was truncated."""
    enc = _encoder_over(tmp_path, 128, status=1)
    enc.start()
    assert _feed(enc)
    with pytest.raises(RuntimeError) as excinfo:
        enc.close()
    message = str(excinfo.value)
    assert "LAST-LINE-OF-STDERR" in message
    assert "omitted" not in message


@requires_ffmpeg
def test_draining_mid_run_does_not_disturb_what_the_child_writes(tmp_path):
    """subprocess hands the child a dup, which shares the file offset.

    Reading with seek() would move where the child's next write lands and
    overwrite its own log, so the drain uses positional reads.
    """
    enc = _encoder_over(tmp_path, 4096, status=0)
    enc.start()
    try:
        # The child writes as soon as it starts, but "as soon as" is not
        # "before this line"; poll rather than race it.
        deadline = time.time() + 30.0
        early = ""
        while time.time() < deadline:
            early = enc._drain_stderr()
            if "LAST-LINE-OF-STDERR" in early:
                break
            time.sleep(0.05)
        assert "LAST-LINE-OF-STDERR" in early
        assert _feed(enc)
    finally:
        enc.close()
    # The child's own bytes are intact: 4096 'e's plus its final line,
    # not a hole where the read repositioned the shared offset.
    assert early.count("e") >= 4096


@requires_ffmpeg
def test_a_real_ffmpeg_failure_still_reports_its_stderr(tmp_path):
    """The capture path has to keep working with actual ffmpeg."""
    enc = FfmpegEncoder(str(tmp_path / "nosuchdir" / "out.mp4"), 64, 48, fps=30)
    enc.start()
    frame = numpy.zeros((48, 64, 4), numpy.uint8)
    with pytest.raises(RuntimeError) as excinfo:
        for _ in range(30):
            enc.write_frame(frame)
        enc.close()
    message = str(excinfo.value)
    assert "nosuchdir" in message or "No such file" in message, message


def test_the_drain_falls_back_when_pread_is_unavailable(tmp_path, monkeypatch):
    """os.pread is Unix-only; the drain must still read on a host without it.

    Recording is X11-only in practice, so this path is for completeness
    rather than a supported platform — but silently returning nothing
    would turn a real ffmpeg error into an empty message.
    """
    enc = FfmpegEncoder(str(tmp_path / "out.mp4"), 64, 48, fps=30)
    handle = tempfile.TemporaryFile()
    handle.write(b"e" * 10 + b"\nWHY-FFMPEG-DIED\n")
    handle.flush()
    enc._stderr_file = handle
    try:
        monkeypatch.delattr(os, "pread", raising=False)
        assert "WHY-FFMPEG-DIED" in enc._drain_stderr()
    finally:
        handle.close()


def test_the_drain_is_quiet_before_the_encoder_starts(tmp_path):
    """No file yet is not an error — write_frame() can be reached first."""
    enc = FfmpegEncoder(str(tmp_path / "out.mp4"), 64, 48, fps=30)
    assert enc._drain_stderr() == ""


# -------- drawtext: text the user typed must be the text on screen --------
#
# Two bugs lived here, and both were invisible in a filter-string
# assertion — you have to render to see them.
#
# An apostrophe cannot be backslash-escaped in a drawtext text= value. A
# filter option inside a filtergraph is unescaped twice (once splitting
# the graph, once splitting a filter's arguments) and a single-quoted
# section has no escape mechanism at all. \' was consumed on the way in,
# so --title "Don't stop" rendered "Dont stop"; with the trailing
# enable= option present the stray quote ran the parse off the end and
# ffmpeg rejected the whole filtergraph with "Filter not found" — no
# recording at all.
#
# And with drawtext's default expansion=normal, a literal % blanks the
# *entire* text. No escaping helps: '100\% done', '100%% done' and a
# verbatim textfile= all render nothing. Only expansion=none renders it,
# which also stops the text being reinterpreted as %{...} directives.

_DRAWTEXT_CASES = [
    "Release demo",
    "Don't stop",
    "100% done",
    "step 1: begin",
    "C:\\path\\to",
    "%{pts}",
    "don't: 50%",
    "it's a 'quote'",
    "a,b[c]d;e",
    "trailing'",
    "'leading",
    "''",
    "50% — done",
]


def _render_gray(vf, width=900, height=120):
    """One frame through `vf` over black, as a gray numpy array."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "color=black:s=%dx%d:d=0.1" % (width, height),
         "-vf", vf, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None, proc.stderr.decode("utf-8", "replace").strip()
    return numpy.frombuffer(proc.stdout, dtype=numpy.uint8).reshape(height, width), ""


def _reference_vf(text, path, font):
    """The same drawtext, with the text supplied through textfile=.

    textfile= takes its content verbatim — no escaping layer at all — so
    this is ground truth for what the user's string should look like.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return (
        "drawtext=fontfile={font}:textfile={tf}:expansion=none:"
        "fontcolor=white:fontsize=40:"
        "box=1:boxcolor=black@0.55:boxborderw=12:x=(w-text_w)/2:y=30:"
        "enable='lt(t,3.0)'".format(
            font=encoder_mod._escape_drawtext(font),
            tf=encoder_mod._escape_drawtext(path),
        )
    )


@requires_ffmpeg
@pytest.mark.parametrize("title", _DRAWTEXT_CASES)
def test_a_title_renders_exactly_as_typed(title, tmp_path):
    font = encoder_mod._find_font()
    if font is None:
        pytest.skip("no usable font on this host")

    ours, our_err = _render_gray(encoder_mod._build_drawtext_filter(title=title))
    assert ours is not None, "ffmpeg rejected the filter for {!r}: {}".format(
        title, our_err
    )

    want, ref_err = _render_gray(
        _reference_vf(title, str(tmp_path / "ref.txt"), font)
    )
    assert want is not None, "the reference render failed: " + ref_err

    # Blank-vs-blank would compare equal and prove nothing — that is how
    # the percent bug hid behind the apostrophe one.
    assert (want > 40).sum() > 0, "the reference rendered nothing for {!r}".format(title)
    assert numpy.array_equal(ours, want), (
        "{!r} does not render as typed: {} lit pixels vs {} in the reference"
        .format(title, int((ours > 40).sum()), int((want > 40).sum()))
    )


def test_an_apostrophe_is_closed_escaped_and_reopened():
    """The only encoding that survives both unescaping passes."""
    quoted = encoder_mod._quote_drawtext_text("Don't")
    assert quoted == "'Don'" + "\\" * 3 + "''t'"
    # and the plain backslash-escape that used to be emitted is not it
    assert "\\'t" not in quoted.replace("\\" * 3 + "'", "")


def test_the_other_specials_keep_their_single_backslash():
    q = encoder_mod._quote_drawtext_text
    assert q("a:b") == "'a\\:b'"
    assert q("100%") == "'100\\%'"
    assert q("a\\b") == "'a\\\\b'"
    # commas and brackets need nothing — the quotes already cover them
    assert q("a,b[c]") == "'a,b[c]'"


def test_every_drawtext_filter_disables_expansion(tmp_path):
    """A % anywhere in the text blanks the whole render without this.

    The font is passed explicitly: both builders return None when they
    cannot find one, and a runner without DejaVu installed would turn
    this assertion into an AttributeError rather than a useful failure.
    """
    font = tmp_path / "fake.ttf"
    font.write_bytes(b"")
    vf = encoder_mod._build_drawtext_filter(
        title="t", overlay_text="o", font_path=str(font)
    )
    assert vf.count("expansion=none") == 2, vf
    chain = subtitles_mod.build_subtitle_filters(
        [Subtitle(text="hello", start=0.0, end=1.0)],
        SubtitleStyle(font_path=str(font)),
    )
    assert "expansion=none" in chain, chain


@requires_ffmpeg
def test_a_subtitle_renders_its_apostrophe(tmp_path):
    """subtitles.py builds its own drawtext chain and shared the bug."""
    font = encoder_mod._find_font()
    if font is None:
        pytest.skip("no usable font on this host")
    style = SubtitleStyle(font_path=font)
    chain = subtitles_mod.build_subtitle_filters(
        [Subtitle(text="don't stop", start=0.0, end=5.0)], style
    )
    got, err = _render_gray(chain)
    assert got is not None, "ffmpeg rejected the subtitle chain: " + err
    plain = subtitles_mod.build_subtitle_filters(
        [Subtitle(text="dont stop", start=0.0, end=5.0)], style
    )
    bare, _ = _render_gray(plain)
    assert bare is not None
    # The apostrophe has to actually be drawn — dropping it silently was
    # the original symptom, and that renders the same as "dont stop".
    assert not numpy.array_equal(got, bare), "the apostrophe was dropped"


# -------- what the CLI claims it wrote --------
#
# Ctrl-C during the countdown stops the recorder before ffmpeg is ever
# started. The recorder says so — it returns a zero stats dict, and its
# own comment reads "Cancelled before the first frame ... there's no
# file" — but the CLI printed the ordinary summary anyway:
#
#   $ fastgrab-record --region 0,0,320,240 --countdown 5 -o out.mp4
#   ^C
#   wrote out.mp4: 0 frames in 0.00s (0.0 fps achieved)     # exit 0
#   $ ls out.mp4
#   ls: cannot access 'out.mp4': No such file or directory
#
# Verified end to end by SIGINTing the real CLI under xvfb.


class _StubRecorder:
    """Stands in for Recorder, returning a chosen stats dict."""

    def __init__(self, stats):
        self._stats = stats

    def record(self, **kwargs):
        return self._stats


def _run_cli_with(monkeypatch, stats, tmp_path, extra=None):
    monkeypatch.setattr(
        recording_cli, "Recorder", lambda **kw: _StubRecorder(stats)
    )
    argv = ["--region", "0,0,64,48", "-o", str(tmp_path / "out.mp4")]
    return recording_cli.main(argv + (extra or []))


def _stats(output, frames=0, written=0, encoder_started=True):
    return {
        "frames": frames,
        "written_frames": written,
        "elapsed_seconds": 0.0 if not frames else 1.0,
        "achieved_fps": 0.0 if not frames else float(frames),
        "output": str(output),
        "encoder_started": encoder_started,
    }


def test_a_cancelled_recording_does_not_claim_a_file(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out.mp4"
    rc = _run_cli_with(
        monkeypatch, _stats(out, encoder_started=False), tmp_path
    )
    captured = capsys.readouterr()
    assert rc == 0, "a deliberate cancel is not an error"
    assert "wrote" not in captured.out, captured.out
    assert "cancelled before the first frame" in captured.err
    assert str(out) in captured.err
    assert not out.exists()


def test_a_recording_that_produced_no_file_is_an_error(monkeypatch, tmp_path, capsys):
    """Frames encoded, ffmpeg happy, nothing on disk — do not say "wrote"."""
    out = tmp_path / "out.mp4"
    rc = _run_cli_with(monkeypatch, _stats(out, frames=10, written=10), tmp_path)
    captured = capsys.readouterr()
    assert rc == 1
    assert "wrote" not in captured.out, captured.out
    assert "does not exist" in captured.err


def test_a_real_recording_still_reports_what_it_wrote(monkeypatch, tmp_path, capsys):
    """The ordinary path must be untouched."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"not really an mp4, but it exists")
    rc = _run_cli_with(monkeypatch, _stats(out, frames=30, written=30), tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    assert "wrote {}".format(out) in captured.out
    assert "30 frames" in captured.out


def test_duplicated_frames_are_still_reported(monkeypatch, tmp_path, capsys):
    """written_frames > frames is the slow-capture case, not a cancel."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    rc = _run_cli_with(monkeypatch, _stats(out, frames=10, written=25), tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    assert "15 duplicated" in captured.out, captured.out


# -------- the clip must last as long as the recording did --------
#
# ffmpeg stamps incoming raw frames at the fixed target rate, so a
# capture loop that cannot keep up would produce a clip shorter than the
# recording and played back too fast, with every subtitle window drifting
# out of place. Recorder guards against that by writing the current frame
# once per elapsed tick.
#
# The guard works — measured against real ffmpeg at up to 6x sustained
# lag, output duration tracked wall-clock to within 15 ms — but nothing
# pinned it. The only assertion on the mechanism was
# `written_frames >= frames`, which is true however badly the pacing
# behaves, so removing the duplication entirely kept the suite green.
#
# These drive recorder.py against a fake clock rather than sleeping. The
# property is arithmetic — how many ticks fell inside the elapsed time —
# and measuring it with real time made it a test of the runner's timer
# granularity instead, which is what it failed on for Windows and macOS.


class _FakeClock:
    """Stands in for the time module inside recorder.py.

    Nothing sleeps: sleep() just moves the clock forward, so a recording
    of any length runs instantly and every tick lands exactly where the
    arithmetic says it should.
    """

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if seconds > 0:
            self.now += seconds


class _ScriptedGrab:
    """A Screenshot stand-in that costs a known amount of clock time."""

    def __init__(self, clock, cost=0.0, stall_at=None, stall_for=0.0):
        self.clock = clock
        self.cost = cost
        self.stall_at = stall_at
        self.stall_for = stall_for
        self.calls = 0
        self.screensize = (64, 48)

    def capture(self, bbox=None):
        self.calls += 1
        if self.stall_at is not None and self.calls == self.stall_at:
            self.clock.now += self.stall_for
        else:
            self.clock.now += self.cost
        return numpy.zeros((48, 64, 4), numpy.uint8)


class _CountingEncoder:
    """Accepts frames and counts them; no ffmpeg involved."""

    def __init__(self, *args, **kwargs):
        self.frames = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_frame(self, frame):
        self.frames += 1


def _paced_recorder(monkeypatch, tmp_path, fps, encoder=None, **grab):
    """A Recorder with its clock, its capture and its encoder all stubbed.

    Screenshot is replaced before construction, not after: Recorder builds
    one in __init__, and the default x11 backend imports the C extension,
    which does not exist on the Windows and macOS runners.
    """
    clock = _FakeClock()
    stub = _ScriptedGrab(clock, **grab)
    monkeypatch.setattr(recorder_mod, "time", clock)
    monkeypatch.setattr(recorder_mod, "Screenshot", lambda **kw: stub)
    monkeypatch.setattr(
        recorder_mod, "FfmpegEncoder", encoder or _CountingEncoder
    )
    rec = Recorder(str(tmp_path / "out.mp4"), bbox=(0, 0, 64, 48), fps=fps)
    return rec, stub


def _assert_tracks_wall_clock(stats, fps, tolerance=1):
    """The clip's length, in frames, must match the time that passed."""
    expected = stats["elapsed_seconds"] * fps
    assert abs(stats["written_frames"] - expected) <= tolerance, (
        "clip is {:.2f}s of video for {:.2f}s of recording ({} frames, "
        "expected about {:.0f})".format(
            stats["written_frames"] / float(fps), stats["elapsed_seconds"],
            stats["written_frames"], expected,
        )
    )


def test_a_capture_that_keeps_up_writes_one_frame_per_tick(monkeypatch, tmp_path):
    rec, _ = _paced_recorder(monkeypatch, tmp_path, 20, cost=0.01)
    stats = rec.record(duration=1.0)
    _assert_tracks_wall_clock(stats, 20)
    assert stats["written_frames"] == stats["frames"], (
        "a capture comfortably inside the tick should need no duplicates"
    )


def test_a_capture_four_times_too_slow_still_fills_the_clip(monkeypatch, tmp_path):
    """The case the duplication exists for: the loop cannot keep up."""
    rec, _ = _paced_recorder(monkeypatch, tmp_path, 20, cost=0.20)
    stats = rec.record(duration=1.0)
    _assert_tracks_wall_clock(stats, 20)
    # A quarter of the target rate: about five captures for twenty ticks.
    assert stats["frames"] <= 6, stats
    assert stats["written_frames"] - stats["frames"] >= 12, stats


def test_a_single_long_stall_is_filled_with_duplicates(monkeypatch, tmp_path):
    """One freeze, not sustained lag — the gap still has to be covered."""
    rec, _ = _paced_recorder(
        monkeypatch, tmp_path, 20, cost=0.01, stall_at=3, stall_for=0.5
    )
    stats = rec.record(duration=1.0)
    _assert_tracks_wall_clock(stats, 20)
    # The freeze alone spans ten ticks at 20 fps and only one frame was
    # captured across it, so at least nine writes must be duplicates.
    assert stats["written_frames"] - stats["frames"] >= 9, stats


def test_stopping_early_also_tracks_wall_clock(monkeypatch, tmp_path):
    """stop_event is how an interactive recording ends, not a deadline.

    The capture costs three ticks so this exercises the duplication too;
    at one tick the loop keeps up and the test would still pass with the
    mechanism removed.
    """
    rec, grab = _paced_recorder(monkeypatch, tmp_path, 20, cost=0.15)
    stop = threading.Event()

    # Fires on the fake clock, not a real timer: stop once the recording
    # has covered about a second of clock time.
    original = grab.capture

    def capture(bbox=None):
        frame = original(bbox)
        if grab.clock.now >= 1001.0:
            stop.set()
        return frame

    grab.capture = capture
    stats = rec.record(stop_event=stop)
    _assert_tracks_wall_clock(stats, 20)
    assert stats["written_frames"] > stats["frames"], stats


def test_every_written_frame_reached_the_encoder(monkeypatch, tmp_path):
    """written_frames is a claim about ffmpeg's input; check it is true."""
    made = []

    class _Recording(_CountingEncoder):
        def __init__(self, *a, **kw):
            _CountingEncoder.__init__(self, *a, **kw)
            made.append(self)

    rec, _ = _paced_recorder(
        monkeypatch, tmp_path, 20, encoder=_Recording, cost=0.1
    )
    stats = rec.record(duration=1.0)
    assert len(made) == 1
    assert made[0].frames == stats["written_frames"]


# -------- close() must say why ffmpeg stopped --------
#
# The timeout path raised "ffmpeg did not exit within 30s" and nothing
# else. Its finally block computed the stderr and then dropped it on the
# floor, so the one failure where ffmpeg's own words matter most — it
# hung, and you cannot ask it anything afterwards because it gets killed
# — was the one failure that arrived with no words at all.

_HANGING_CHILD = (
    "import sys, time\n"
    "sys.stderr.buffer.write(b'WHY-IT-HUNG\\n')\n"
    "sys.stderr.buffer.flush()\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)

_SILENT_HANGING_CHILD = (
    "import sys, time\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)

_DYING_CHILD = (
    "import sys\n"
    "sys.stderr.buffer.write(b'DIED-BECAUSE-OF-THIS\\n')\n"
    "sys.stderr.buffer.flush()\n"
    "sys.exit(3)\n"
)


def _encoder_running(tmp_path, source):
    enc = FfmpegEncoder(str(tmp_path / "out.mp4"), 64, 48, fps=30)
    enc._build_argv = lambda *a, **k: [sys.executable, "-c", source]
    enc.start()
    return enc


@requires_ffmpeg
def test_a_hung_encoder_reports_what_it_said_before_hanging(tmp_path):
    enc = _encoder_running(tmp_path, _HANGING_CHILD)
    with pytest.raises(RuntimeError) as excinfo:
        enc.close(timeout=1.0)
    message = str(excinfo.value)
    assert "did not exit within 1.0s" in message, message
    assert "WHY-IT-HUNG" in message, message


@requires_ffmpeg
def test_a_hung_encoder_that_said_nothing_says_so(tmp_path):
    """An empty tail must not trail off the end of the message."""
    enc = _encoder_running(tmp_path, _SILENT_HANGING_CHILD)
    with pytest.raises(RuntimeError) as excinfo:
        enc.close(timeout=1.0)
    message = str(excinfo.value)
    assert "did not exit within 1.0s" in message, message
    assert "wrote nothing to stderr" in message, message


@requires_ffmpeg
def test_a_hung_encoder_is_killed_and_reaped(tmp_path):
    """The timeout must not leave the process behind."""
    enc = _encoder_running(tmp_path, _HANGING_CHILD)
    proc = enc._proc
    with pytest.raises(RuntimeError):
        enc.close(timeout=1.0)
    assert proc.poll() is not None, "the hung encoder outlived close()"
    assert enc._proc is None
    assert enc._stderr_file is None, "the stderr file was not released"


@requires_ffmpeg
def test_a_broken_pipe_on_close_does_not_hide_the_real_cause(tmp_path):
    """ffmpeg died first, so closing its stdin fails.

    Reporting that BrokenPipeError names a symptom and nothing else, and
    it skipped the exit status and stderr that say what actually
    happened.
    """
    enc = _encoder_running(tmp_path, _DYING_CHILD)
    enc._proc.wait()

    def boom():
        raise BrokenPipeError(32, "Broken pipe")

    enc._proc.stdin.close = boom
    with pytest.raises(RuntimeError) as excinfo:
        enc.close(timeout=5.0)
    message = str(excinfo.value)
    assert "status 3" in message, message
    assert "DIED-BECAUSE-OF-THIS" in message, message


@requires_ffmpeg
def test_a_clean_close_still_raises_nothing(tmp_path):
    """The ordinary path must be untouched."""
    enc = _encoder_running(tmp_path, "import sys\nsys.stdin.buffer.read()\n")
    enc.close(timeout=10.0)
    assert enc._proc is None
    assert enc._stderr_file is None


# -------- two regressions the first version of this check introduced --------
#
# Both found by review of the merged change, and both confirmed by
# running ffmpeg rather than by reading the code.

def test_zero_frames_with_a_started_encoder_is_not_a_cancellation(
    monkeypatch, tmp_path, capsys
):
    """A frame count of nought does not mean nothing was produced.

    If the loop stops after ffmpeg starts but before the first capture,
    ffmpeg writes and closes an empty container quite happily — measured
    at 261 bytes for mp4 and 465 for webm, both with a clean exit. The
    first version of this branch keyed on the frame count and so
    announced that a file which exists "was not written", which is the
    same lie it was added to prevent, pointing the other way.
    """
    out = tmp_path / "out.mp4"
    out.write_bytes(b"\0" * 261)  # what ffmpeg leaves behind
    rc = _run_cli_with(
        monkeypatch, _stats(out, frames=0, written=0, encoder_started=True),
        tmp_path,
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "was not written" not in captured.err, captured.err
    assert "wrote {}".format(out) in captured.out, captured.out


def test_a_file_url_output_is_resolved_before_looking_for_it(
    monkeypatch, tmp_path, capsys
):
    """`-o file:out.mp4` writes out.mp4; the check must not fail it.

    ffmpeg's file: protocol exists so a name containing a colon, or one
    starting with a dash, can be given unambiguously. Confirmed against
    real ffmpeg: `file:/tmp/url.mp4` produced /tmp/url.mp4 and exited
    cleanly, while os.path.exists on the raw string was False — so the
    first version of this check failed a recording that had worked.
    """
    real = tmp_path / "out.mp4"
    real.write_bytes(b"a real recording")
    rc = _run_cli_with(
        monkeypatch, _stats("file:" + str(real), frames=10, written=10),
        tmp_path,
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "does not exist" not in captured.err
    assert "wrote file:{}".format(real) in captured.out, captured.out


def test_a_non_file_destination_is_not_checked_on_disk(
    monkeypatch, tmp_path, capsys
):
    """ffmpeg can write to a protocol URL; there is no path to stat."""
    rc = _run_cli_with(
        monkeypatch, _stats("rtmp://example.invalid/live/x.mp4",
                            frames=10, written=10),
        tmp_path,
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "does not exist" not in captured.err


def test_a_genuinely_missing_local_file_is_still_an_error(
    monkeypatch, tmp_path, capsys
):
    """The check must keep working for the ordinary case."""
    out = tmp_path / "gone.mp4"
    rc = _run_cli_with(monkeypatch, _stats(out, frames=10, written=10), tmp_path)
    captured = capsys.readouterr()
    assert rc == 1
    assert "does not exist" in captured.err


def test_a_missing_file_url_target_is_reported_by_its_real_path(
    monkeypatch, tmp_path, capsys
):
    """Resolving must not turn a real failure into a pass."""
    out = tmp_path / "gone.mp4"
    rc = _run_cli_with(
        monkeypatch, _stats("file:" + str(out), frames=10, written=10), tmp_path
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert str(out) in captured.err
    assert "file:" not in captured.err, "report the path ffmpeg writes, not the URL"


# -------- Ctrl-C outside the recording loop --------
#
# The loop installs its own SIGINT handler so a stop finalises the
# container. Everything around it did not behave:
#
#   * before the handler is installed — argument parsing, the region
#     selector, the config dialog — Ctrl-C ended the command with a bare
#     KeyboardInterrupt traceback. The selector is the case that matters:
#     it can sit open for as long as the user takes to drag a box.
#     (One window stays: the ~0.22s of module import, measured, which
#     happens before main() is called and so before any code here can
#     catch anything. A SIGINT at 0.2s still ends in a traceback; at 0.6s
#     it does not.)
#   * after record() returns, the handler was still installed, so a
#     Ctrl-C during the summary set an event nothing reads any more and
#     the command could not be interrupted at all.
#   * main() is importable and reachable as a library call, and it left
#     the process's signal disposition permanently changed.
#
# Installing the handler earlier would fix only the first, and would buy
# it by making the interactive setup ignore Ctrl-C, which is worse.


def test_an_interrupt_before_recording_exits_cleanly(monkeypatch, tmp_path, capsys):
    """No traceback, and the shell's conventional status for SIGINT."""

    def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(recording_cli, "Recorder", interrupted)
    try:
        rc = recording_cli.main(
            ["--region", "0,0,64,48", "-o", str(tmp_path / "out.mp4")]
        )
    except KeyboardInterrupt:
        # Caught here on purpose. pytest treats a KeyboardInterrupt as a
        # request to abandon the session, so letting one escape would
        # abort the whole run at whatever point this test happens to sit
        # rather than reporting which behaviour regressed.
        pytest.fail("main() let the KeyboardInterrupt escape to the caller")
    captured = capsys.readouterr()
    assert rc == 130, "130 is what bash reports for a child killed by SIGINT"
    assert "fastgrab: interrupted" in captured.err
    assert "Traceback" not in captured.err


def test_the_signal_handlers_are_restored_after_recording(monkeypatch, tmp_path):
    """Left installed, they swallow a Ctrl-C during the summary."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))

    rc = _run_cli_with(monkeypatch, _stats(out, frames=5, written=5), tmp_path)

    assert rc == 0
    assert (signal.getsignal(signal.SIGINT),
            signal.getsignal(signal.SIGTERM)) == before


def test_the_signal_handlers_are_restored_when_recording_fails(
    monkeypatch, tmp_path
):
    """The restore has to survive the error path too."""
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))

    class _Failing:
        def record(self, **kwargs):
            raise RuntimeError("ffmpeg fell over")

    monkeypatch.setattr(recording_cli, "Recorder", lambda **kw: _Failing())
    rc = recording_cli.main(
        ["--region", "0,0,64,48", "-o", str(tmp_path / "out.mp4")]
    )

    assert rc == 1
    assert (signal.getsignal(signal.SIGINT),
            signal.getsignal(signal.SIGTERM)) == before


def test_the_handler_is_installed_while_recording(monkeypatch, tmp_path):
    """The restore must not undo the thing it is restoring around.

    A Ctrl-C mid-recording still has to set the stop event rather than
    raise, or ffmpeg loses the chance to write its trailer.
    """
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    seen = {}

    class _Watching:
        def record(self, stop_event=None, **kwargs):
            seen["handler"] = signal.getsignal(signal.SIGINT)
            seen["default"] = signal.default_int_handler
            # what a real SIGINT would do at this moment
            seen["handler"](signal.SIGINT, None)
            seen["stopped"] = stop_event.is_set()
            return _stats(out, frames=5, written=5)

    monkeypatch.setattr(recording_cli, "Recorder", lambda **kw: _Watching())
    rc = recording_cli.main(
        ["--region", "0,0,64,48", "-o", str(out)]
    )
    assert rc == 0
    assert seen["handler"] is not seen["default"], "the loop ran unprotected"
    assert seen["stopped"], "a Ctrl-C mid-recording did not ask it to stop"


# -------- the sidecar must never destroy anything --------
#
# Reproduced before it was fixed: with the DEFAULT drawtext backend,
# FfmpegEncoder(subtitle_sidecar=<an existing video>) followed by
# _build_argv() truncated a 2500-byte file to a 678-byte ASS script. No
# ffmpeg process, no capture, no opt-in to ASS at all. These pin the
# properties that stop it, none of which a rendering test can see.


def _sidecar_encoder(tmp_path, sidecar, output="clip.mp4", **kw):
    return FfmpegEncoder(
        str(tmp_path / output), 64, 48, fps=10,
        subtitles=[Subtitle(text="sub", start=0.0, end=1.0)],
        subtitle_sidecar=str(sidecar), **kw
    )


def test_building_argv_never_touches_the_sidecar_path(tmp_path):
    """The exact shape of the original bug."""
    victim = tmp_path / "precious.mp4"
    victim.write_bytes(b"A REAL VIDEO" * 200)
    before = victim.read_bytes()

    enc = _sidecar_encoder(tmp_path, victim)
    enc._build_argv()
    enc._build_argv()          # twice, in case the first was the only write

    assert victim.read_bytes() == before, "argv construction rewrote the file"


def test_a_sidecar_that_is_the_output_file_is_refused(tmp_path):
    """The one overwrite that is never what anyone meant."""
    out = tmp_path / "clip.mp4"
    with pytest.raises(ValueError, match="output file"):
        FfmpegEncoder(
            str(out), 64, 48, fps=10,
            subtitles=[Subtitle(text="s", start=0.0, end=1.0)],
            subtitle_sidecar=str(out),
        )


def test_the_output_alias_check_does_not_need_the_file_to_exist(tmp_path):
    """Refused before the recording, not after it has overwritten itself."""
    out = tmp_path / "not-yet.mp4"
    assert not out.exists()
    with pytest.raises(ValueError, match="output file"):
        FfmpegEncoder(
            str(out), 64, 48, fps=10,
            subtitles=[Subtitle(text="s", start=0.0, end=1.0)],
            subtitle_sidecar="file:" + str(out),
        )


def test_a_symlinked_alias_is_still_the_output_file(tmp_path):
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"video")
    link = tmp_path / "alias.mp4"
    link.symlink_to(out)
    with pytest.raises(ValueError, match="output file"):
        _sidecar_encoder(tmp_path, link)


def test_a_sidecar_without_subtitles_is_refused(tmp_path):
    """Otherwise it silently produces nothing at all."""
    with pytest.raises(ValueError, match="no subtitles"):
        FfmpegEncoder(
            str(tmp_path / "clip.mp4"), 64, 48, fps=10,
            subtitle_sidecar=str(tmp_path / "clip.ass"),
        )


def test_a_remote_sidecar_destination_is_refused(tmp_path):
    """There is nothing local to write, and ffmpeg will not do it for us."""
    with pytest.raises(ValueError, match="local path"):
        _sidecar_encoder(tmp_path, "rtmp://example.invalid/live/clip.ass")


@requires_ffmpeg
def test_a_failed_encode_leaves_an_existing_sidecar_alone(tmp_path):
    """Publishing after success means a failure changes nothing on disk."""
    sidecar = tmp_path / "clip.ass"
    sidecar.write_text("MY EARLIER CAPTIONS", encoding="utf-8")

    enc = _sidecar_encoder(tmp_path, sidecar)
    enc._build_argv = lambda *a, **k: [sys.executable, "-c",
                                       "import sys; sys.stdin.buffer.read();"
                                       " sys.exit(4)"]
    enc.start()
    with pytest.raises(RuntimeError, match="status 4"):
        enc.close()
    assert sidecar.read_text(encoding="utf-8") == "MY EARLIER CAPTIONS"


def test_the_script_ffmpeg_reads_is_never_the_callers_sidecar(tmp_path):
    """ffmpeg opens the script lazily, so it must own a private copy.

    Deleting the script after Popen returns but before the first frame
    still fails ASS initialisation — a successful Popen does not mean it
    has been read. Two recordings pointed at one sidecar would otherwise
    be able to consume each other's.
    """
    sidecar = tmp_path / "clip.ass"
    enc = _sidecar_encoder(tmp_path, sidecar, subtitle_backend="ass")
    script = enc._write_ass(enc.ass_document())
    argv = enc._build_argv(script)
    vf_value = argv[argv.index("-vf") + 1]
    assert str(sidecar) not in vf_value, "ffmpeg was pointed at the caller's file"
    assert script != str(sidecar)
    enc.close()


def test_the_temporary_script_goes_away_when_start_fails(tmp_path):
    enc = _sidecar_encoder(tmp_path, tmp_path / "clip.ass",
                           subtitle_backend="ass")
    enc._build_argv = lambda *a, **k: ["/nonexistent-binary-for-fastgrab"]
    with pytest.raises(Exception):
        enc.start()
    assert enc._ass_path is None
    assert enc._stderr_file is None


@requires_ffmpeg
@pytest.mark.parametrize("name", [
    "it's.ttf", "a:b.ttf", "a,b.ttf", "a[b].ttf",
    pytest.param("a\\b.ttf", marks=pytest.mark.skipif(
        os.name == "nt",
        reason="a backslash is a path separator on Windows, not a filename",
    )),
])
def test_a_font_path_with_awkward_characters_still_renders(name, tmp_path):
    """The real oracle: the same font under an awkward name must render
    identically to the same font under a plain one.

    Measured before the fix: apostrophe, colon, comma and bracket each
    made ffmpeg reject the filtergraph outright, and backslash produced
    1191 lit pixels against the reference's 1920.
    """
    base = encoder_mod._find_font()
    if base is None:
        pytest.skip("no usable font on this host")

    reference, err = _render_gray(
        encoder_mod._build_drawtext_filter(title="Demo", font_path=base)
    )
    assert reference is not None, err
    assert (reference > 40).sum() > 0, "the reference rendered nothing"

    awkward = tmp_path / name
    shutil.copy(base, awkward)
    got, err = _render_gray(
        encoder_mod._build_drawtext_filter(title="Demo", font_path=str(awkward))
    )
    assert got is not None, "ffmpeg rejected the filter for {!r}: {}".format(
        name, err
    )
    assert numpy.array_equal(got, reference), (
        "{!r} rendered differently: {} lit pixels vs {}".format(
            name, int((got > 40).sum()), int((reference > 40).sum())
        )
    )


def test_publishing_refuses_a_sidecar_that_became_the_recording(tmp_path):
    """The publish-time guard, driven directly.

    The construction check cannot answer this: when neither file exists,
    samefile() raises and comparing realpath strings says two names are
    two files. They may not be -- a case-insensitive filesystem is the
    real case -- and by publish time that file holds the recording.

    Driven straight at _publish_sidecar rather than through a fake
    ffmpeg. The first version of this test ran a child that wrote the
    output only after close() sent it EOF, so the alias was never
    actually created and the test passed against the *unguarded* code
    too. A regression test that cannot fail is worse than none.
    """
    out = tmp_path / "clip.mp4"
    sidecar = tmp_path / "captions.ass"
    enc = FfmpegEncoder(
        str(out), 64, 48, fps=10,
        subtitles=[Subtitle(text="s", start=0.0, end=1.0)],
        subtitle_sidecar=str(sidecar),
    )   # accepted: at this point neither path exists

    # Now they are the same file, exactly as the encode finishing would
    # leave them on a filesystem that folds case.
    out.write_bytes(b"THE RECORDING")
    os.link(out, sidecar)

    with pytest.raises(RuntimeError, match="recording that was just written"):
        enc._publish_sidecar("[Script Info]\n")
    assert out.read_bytes() == b"THE RECORDING"


def test_publishing_still_works_when_the_sidecar_is_a_separate_file(tmp_path):
    """The guard must not refuse the ordinary case."""
    out = tmp_path / "clip.mp4"
    sidecar = tmp_path / "captions.ass"
    enc = FfmpegEncoder(
        str(out), 64, 48, fps=10,
        subtitles=[Subtitle(text="s", start=0.0, end=1.0)],
        subtitle_sidecar=str(sidecar),
    )
    out.write_bytes(b"THE RECORDING")
    enc._publish_sidecar("[Script Info]\n")
    assert sidecar.read_text(encoding="utf-8") == "[Script Info]\n"
    assert out.read_bytes() == b"THE RECORDING"


# -------- a literal backslash in a subtitle must stay one line --------
#
# ASS reads \n, \N and \h as a soft break, a hard break and a
# non-breaking space, so a subtitle carrying a literal one has to be
# defused. The branch doubled the backslash; measured through libass that
# does not work -- "left\Nright" still rendered as two lines, corrupting
# both the burned-in video and the exported sidecar.

_ASS_HEAD = (
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 120\n\n"
    "[V4+ Styles]\n"
    "Format: Name,Fontname,Fontsize,PrimaryColour,Alignment,MarginV\n"
    "Style: D,DejaVu Sans,36,&H00FFFFFF,2,10\n\n"
    "[Events]\n"
    "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
)


def _render_ass_line(text_field, tmp_path):
    """Render one Dialogue line through libass; return (ink, height, width)."""
    script = tmp_path / "line.ass"
    script.write_text(
        _ASS_HEAD + "Dialogue: 0,0:00:00.00,0:00:05.00,D,,0,0,0,,%s\n"
        % text_field, encoding="utf-8")
    frame, err = _render_gray(
        "ass=filename=" + encoder_mod._escape_filter_path(str(script)),
        width=640, height=120,
    )
    if frame is None:
        return None, err
    ys, xs = numpy.nonzero(frame > 40)
    if not len(ys):
        return (0, 0, 0), ""
    return (int((frame > 40).sum()),
            int(ys.max() - ys.min() + 1),
            int(xs.max() - xs.min() + 1)), ""


@requires_ffmpeg
@pytest.mark.parametrize("letter", ["n", "N", "h"])
def test_a_literal_backslash_before_a_control_letter_stays_on_one_line(
    letter, tmp_path
):
    """The escaped form must render as the plain text plus a backslash.

    Width is the reliable measure: exactly ten pixels wider than the same
    text without the backslash, at this size and font, and on one line.
    """
    plain, _ = _render_ass_line("left" + letter + "right", tmp_path)
    if plain is None or plain[0] == 0:
        pytest.skip("libass rendered nothing here; no usable font")

    escaped_field = subtitles_mod._escape_ass("left\\" + letter + "right")
    escaped, err = _render_ass_line(escaped_field, tmp_path)
    assert escaped is not None, err

    assert escaped[1] == plain[1], (
        "escaped \\{} wrapped onto {} lines' worth of height ({} vs {})"
        .format(letter, escaped[1] / float(plain[1]), escaped[1], plain[1])
    )
    # The backslash's width is measured with whatever font libass
    # actually resolved, not hardcoded: DejaVu Sans is 10px here and
    # Liberation Mono, which libass substitutes when DejaVu is missing,
    # is 18. A fixed number turns this into a test of the host's fonts.
    with_inert, _ = _render_ass_line("left\\zright", tmp_path)
    without, _ = _render_ass_line("leftzright", tmp_path)
    backslash_width = with_inert[2] - without[2]
    assert backslash_width > 0, "could not measure the backslash in this font"

    assert escaped[2] == plain[2] + backslash_width, (
        "expected the plain text plus one backslash ({}px in this font), "
        "got width {} vs {}".format(backslash_width, escaped[2], plain[2])
    )


@requires_ffmpeg
def test_the_zero_width_space_does_not_disturb_a_working_backslash(tmp_path):
    """z is not an ASS tag, so that backslash already rendered correctly.

    Inserting the separator there must change nothing at all — that is
    what makes it safe to insert before n, N and h.
    """
    without, _ = _render_ass_line("left\\zright", tmp_path)
    if without is None or without[0] == 0:
        pytest.skip("libass rendered nothing here; no usable font")
    with_sep, err = _render_ass_line(
        "left\\" + subtitles_mod._ZWSP + "zright", tmp_path)
    assert with_sep is not None, err
    assert with_sep == without


# -------- CLI: deriving and refusing a sidecar --------

def test_a_bare_sidecar_flag_derives_from_the_path_ffmpeg_writes():
    """`-o file:demo.mp4` writes demo.mp4, so the sidecar is demo.ass.

    Deriving from the string as typed produced the literal filename
    "file:demo.ass".
    """
    assert recording_cli._resolve_sidecar("", "demo.mp4") == "demo.ass"
    assert recording_cli._resolve_sidecar("", "file:demo.mp4") == "demo.ass"
    assert recording_cli._resolve_sidecar("", "/tmp/a.b/demo.webm") == "/tmp/a.b/demo.ass"


def test_an_explicit_sidecar_path_is_taken_as_given():
    assert recording_cli._resolve_sidecar("caps.ass", "file:demo.mp4") == "caps.ass"


def test_a_bare_sidecar_flag_needs_a_local_output():
    """There is no name to derive from a stream URL."""
    with pytest.raises(ValueError, match="needs a path of its own"):
        recording_cli._resolve_sidecar("", "rtmp://example.invalid/live/x.mp4")


def test_no_sidecar_flag_stays_none():
    assert recording_cli._resolve_sidecar(None, "demo.mp4") is None


def test_a_refused_sidecar_prints_instead_of_tracebacking(capsys, tmp_path):
    """Both this and Recorder's own validation sit outside the recording
    try/except, so an unroutable refusal used to end in a traceback."""
    rc = recording_cli.main([
        "--region", "0,0,64,48", "-o", "rtmp://example.invalid/live/x.mp4",
        "--subtitle", "0.0-1.0:hi", "--subtitle-sidecar",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err
    assert "needs a path of its own" in captured.err
    assert "Traceback" not in captured.err


def test_the_colour_error_blames_the_name_not_the_format():
    """chartreuse converts fine as 0x7FFF00; the lookup table is the limit.

    Saying it "cannot be converted to ASS" sent the reader looking for a
    format limitation that does not exist.
    """
    with pytest.raises(ValueError) as excinfo:
        subtitles_mod._ass_color("chartreuse")
    message = str(excinfo.value)
    assert "unsupported colour name" in message
    assert "0xRRGGBB" in message
    assert "sidecar" in message, "say that it bites with drawtext too"
    # and the colour itself is expressible, which is the point
    assert subtitles_mod._ass_color("0x7FFF00").startswith("&H")


# -------- ASS cue timing --------

def test_a_cue_too_short_for_a_centisecond_is_refused():
    """It would round to zero length and never be drawn.

    Measured: 1.001-1.004 becomes "0:00:01.00,0:00:01.00" and renders
    nothing at any frame. Writing it anyway means a subtitle missing from
    the video and from the sidecar, with nothing to say why.
    """
    with pytest.raises(ValueError, match="centisecond"):
        subtitles_mod.build_ass_document(
            [Subtitle(text="blink", start=1.001, end=1.004)]
        )


def test_a_cue_of_exactly_one_centisecond_is_fine():
    """The boundary itself must not be refused."""
    doc = subtitles_mod.build_ass_document(
        [Subtitle(text="brief", start=1.00, end=1.01)]
    )
    assert "0:00:01.00,0:00:01.01" in doc


@requires_ffmpeg
def test_the_ass_cue_end_is_exclusive(tmp_path):
    """One frame's difference from drawtext, and the format's own rule.

    Rendered at 100 fps: for a 0.5-1.0 cue both backends draw from 0.50,
    and at exactly 1.00 drawtext still draws while ASS has stopped.
    """
    font = encoder_mod._find_font()
    if font is None:
        pytest.skip("no usable font on this host")
    style = SubtitleStyle(font_path=font)
    subs = [Subtitle(text="HELLO", start=0.5, end=1.0)]

    script = tmp_path / "cue.ass"
    script.write_text(
        subtitles_mod.build_ass_document(subs, style, width=480, height=80),
        encoding="utf-8")
    ass_vf = "ass=filename=" + encoder_mod._escape_filter_path(str(script))
    draw_vf = subtitles_mod.build_subtitle_filters(subs, style)

    def strip(vf):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi",
             "-i", "color=black:s=480x80:d=2:r=100", "-vf", vf,
             "-frames:v", "110", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True)
        assert proc.returncode == 0, proc.stderr.decode()[:200]
        arr = numpy.frombuffer(proc.stdout, dtype=numpy.uint8)
        n = len(arr) // (80 * 480)
        return [(f > 40).sum() for f in arr[: n * 80 * 480].reshape(n, 80, 480)]

    drawn, assed = strip(draw_vf), strip(ass_vf)
    assert drawn[49] == 0 and assed[49] == 0, "drawn before the cue started"
    assert drawn[50] > 0 and assed[50] > 0, "both must start at 0.50"
    assert drawn[99] > 0 and assed[99] > 0, "both must still be up at 0.99"
    assert drawn[100] > 0, "drawtext includes the end instant"
    assert assed[100] == 0, "ASS excludes it — this is the documented gap"
