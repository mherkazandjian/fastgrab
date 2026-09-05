"""Smoke tests for fastgrab.recording.

Pure-numpy overlay tests, filter-assembly tests, and CLI parsing tests
run anywhere. Tests that actually spawn ffmpeg are skipped when ffmpeg
is not on PATH; recorder tests additionally need an X11 DISPLAY — i.e.
inside ``docker compose run --rm test``.
"""
import argparse
import os
import shutil
import subprocess
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
    assert "fontfile=" + encoder_mod._escape_drawtext(str(fake_font)) in vf


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


def test_font_path_is_escaped_in_drawtext():
    # A ':' inside the font path would be read as an option separator by
    # the filter parser, so it has to be escaped like the text is. The
    # path is synthetic (explicit font_path= skips the existence check)
    # because ':' is not a legal filename character on Windows.
    font = "/fonts:odd/fake.ttf"
    escaped = "/fonts\\:odd/fake.ttf"

    vf = encoder_mod._build_drawtext_filter(title="t", font_path=font)
    assert "fontfile=" + escaped + ":" in vf
    assert "fontfile=" + font + ":" not in vf

    subs = [Subtitle(text="x", start=0.0, end=1.0)]
    vf = subtitles_mod.build_subtitle_filters(subs, SubtitleStyle(font_path=font))
    assert "fontfile=" + escaped + ":" in vf


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
    assert esc("C:\\new") == "C:\\\\new"
    assert esc("a\\Nb") == "a\\\\Nb"
    assert esc("a\\hb") == "a\\\\hb"
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
    argv = enc._build_argv()
    vf_value = argv[argv.index("-vf") + 1]
    assert vf_value.startswith("ass=filename=")
    assert "drawtext=" not in vf_value          # ASS replaces the chain
    script = enc._ass_path
    assert script and os.path.exists(script)
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
    argv = enc._build_argv()
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


def test_subtitle_sidecar_is_written_and_kept(tmp_path, monkeypatch):
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
    assert sidecar.exists()
    assert "[Events]" in sidecar.read_text(encoding="utf-8")
    enc.close()
    assert sidecar.exists()      # the caller's file is never deleted
    assert enc._ass_path is None


def test_subtitle_sidecar_that_cannot_be_written_reports_clearly(tmp_path):
    enc = FfmpegEncoder(
        str(tmp_path / "clip.mp4"), 64, 48, fps=10,
        subtitles=[Subtitle(text="sub", start=0.0, end=1.0)],
        subtitle_sidecar=str(tmp_path / "no-such-dir" / "clip.ass"),
    )
    # The CLI only prints RuntimeError/ValueError, so a bad path handed
    # in by the user must not surface as a bare OSError traceback.
    with pytest.raises(RuntimeError, match="subtitle sidecar"):
        enc._build_argv()


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
    import threading

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
def test_recorder_smoke_mp4(tmp_path):
    out = tmp_path / "smoke.mp4"
    rec = Recorder(
        output_path=str(out),
        bbox=(0, 0, 240, 180),
        fps=10,
        backend="x11",
    )
    seen = []
    stats = rec.record(duration=0.6, on_progress=lambda n, _e: seen.append(n))
    assert stats["frames"] >= 4
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
