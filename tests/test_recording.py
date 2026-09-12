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
import threading
import time

import numpy
import pytest

from fastgrab.recording import (
    BlurStyle,
    ClickStyle,
    FfmpegEncoder,
    Recorder,
    Subtitle,
    SubtitleStyle,
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


# --------------------------------------------------------------------
# Blur / redaction wiring
# --------------------------------------------------------------------

class _StubRecorder:
    """Captures the kwargs main() builds without touching a display."""

    last = None

    def __init__(self, **kwargs):
        _StubRecorder.last = kwargs

    def record(self, **_kwargs):
        return {
            "frames": 1, "written_frames": 1, "elapsed_seconds": 1.0,
            "achieved_fps": 1.0, "output": "stub.mp4",
        }


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(recording_cli, "Recorder", _StubRecorder)
    _StubRecorder.last = None
    # main() installs its own SIGINT/SIGTERM handlers so ffmpeg can
    # finalise the container on Ctrl-C. Put the originals back, or the
    # rest of the pytest session runs with Ctrl-C disarmed.
    saved = {sig: signal.getsignal(sig)
             for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        assert recording_cli.main(argv) == 0
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    return _StubRecorder.last


def test_cli_parse_blur_region_allows_single_pixel_sizes():
    assert recording_cli._parse_blur_region("1,2,3,4") == (1, 2, 3, 4)
    assert recording_cli._parse_blur_region("0,0,1,1") == (0, 0, 1, 1)
    for bad in ("1,2,3", "a,b,c,d", "-1,0,10,10", "0,0,0,4"):
        with pytest.raises(argparse.ArgumentTypeError):
            recording_cli._parse_blur_region(bad)


def test_cli_blur_is_repeatable():
    parser = recording_cli.build_parser()
    args = parser.parse_args([
        "--fullscreen", "-o", "x.mp4",
        "--blur", "0,0,10,10", "--blur", "20,20,5,5",
    ])
    assert args.blur == [(0, 0, 10, 10), (20, 20, 5, 5)]


def test_cli_blur_and_blur_all_are_mutually_exclusive():
    parser = recording_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--fullscreen", "-o", "x.mp4", "--blur", "0,0,10,10", "--blur-all",
        ])


def test_cli_blur_tuning_without_a_target_is_an_error(capsys):
    with pytest.raises(SystemExit):
        recording_cli.main(["--fullscreen", "-o", "x.mp4",
                            "--blur-method", "fill"])
    assert "--blur" in capsys.readouterr().err


def test_cli_blur_regions_reach_the_recorder(monkeypatch):
    kwargs = _run_cli(monkeypatch, [
        "--fullscreen", "-o", "x.mp4", "--blur", "10,20,30,40",
    ])
    assert kwargs["blur"] == [(10, 20, 30, 40)]
    # No tuning flags → default style, built by BlurStyle itself.
    assert kwargs["blur_style"] is None


def test_cli_blur_all_becomes_true(monkeypatch):
    kwargs = _run_cli(monkeypatch, ["--fullscreen", "-o", "x.mp4", "--blur-all"])
    assert kwargs["blur"] is True


def test_cli_without_blur_passes_nothing(monkeypatch):
    kwargs = _run_cli(monkeypatch, ["--fullscreen", "-o", "x.mp4"])
    assert kwargs["blur"] is None
    assert kwargs["blur_style"] is None


def test_cli_blur_style_built_from_flags(monkeypatch):
    style = _run_cli(monkeypatch, [
        "--fullscreen", "-o", "x.mp4", "--blur-all",
        "--blur-method", "fill", "--blur-color", "1,2,3",
    ])["blur_style"]
    assert style.method == "fill"
    assert style.color == (1, 2, 3)

    style = _run_cli(monkeypatch, [
        "--fullscreen", "-o", "x.mp4", "--blur", "0,0,80,40",
        "--blur-method", "gaussian", "--blur-radius", "7",
    ])["blur_style"]
    assert style.method == "gaussian"
    assert style.radius == 7

    style = _run_cli(monkeypatch, [
        "--fullscreen", "-o", "x.mp4", "--blur", "0,0,80,40",
        "--blur-method", "pixelate", "--blur-block", "9",
    ])["blur_style"]
    assert style.method == "pixelate"
    assert style.block == 9


@pytest.mark.parametrize("flags,ignored", [
    (["--blur-color", "0,0,0"], "--blur-color"),
    (["--blur-method", "box", "--blur-block", "8"], "--blur-block"),
    (["--blur-method", "fill", "--blur-radius", "4"], "--blur-radius"),
    (["--blur-method", "pixelate", "--blur-radius", "4"], "--blur-radius"),
])
def test_cli_rejects_options_the_method_would_ignore(flags, ignored, capsys):
    """--blur-color with a box blur must not quietly leave a blur behind."""
    with pytest.raises(SystemExit):
        recording_cli.main(["--fullscreen", "-o", "x.mp4", "--blur-all"]
                           + flags)
    err = capsys.readouterr().err
    assert ignored in err
    assert "silently ignored" in err


def test_cli_rejects_a_pixelate_block_that_changes_nothing(capsys):
    with pytest.raises(SystemExit):
        recording_cli.main([
            "--fullscreen", "-o", "x.mp4", "--blur-all",
            "--blur-method", "pixelate", "--blur-block", "1",
        ])
    assert "leaves the region unchanged" in capsys.readouterr().err


def test_cli_rejects_unknown_blur_method():
    parser = recording_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--fullscreen", "-o", "x.mp4",
                           "--blur-all", "--blur-method", "swirl"])


@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_forwards_blur_to_its_screenshot():
    style = BlurStyle(method="fill", color=(1, 2, 3))
    rec = Recorder(
        output_path="/tmp/unused.mp4", bbox=(0, 0, 64, 48), backend="x11",
        blur=[(0, 0, 10, 10)], blur_style=style,
    )
    assert rec._grab.blur == ((0, 0, 10, 10),)
    assert rec._grab.blur_style is style


@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_blur_can_be_changed_after_construction():
    """Regression: rec.blur was a copy, so updates never reached recording."""
    rec = Recorder(
        output_path="/tmp/unused.mp4", bbox=(0, 0, 64, 48), backend="x11",
    )
    assert rec.blur is None

    rec.blur = [(0, 0, 16, 16)]
    rec.blur_style = BlurStyle(method="fill", color=(7, 11, 13))
    assert rec._grab.blur == ((0, 0, 16, 16),)
    assert rec._grab.blur_style is rec.blur_style

    frame = rec._grab.capture(bbox=(0, 0, 64, 48))
    assert (frame[0:16, 0:16, 0] == 7).all()

    # And both setters validate, like the constructor does.
    with pytest.raises(ValueError):
        rec.blur = [(0, 0, 0, 16)]
    with pytest.raises(TypeError):
        rec.blur_style = "fill"


@pytest.mark.skipif(not _x11_available(), reason="needs X11 DISPLAY")
def test_recorder_frames_are_redacted_before_overlays():
    """The frame handed to the encoder must already be blurred."""
    rec = Recorder(
        output_path="/tmp/unused.mp4", bbox=(0, 0, 64, 48), backend="x11",
        blur=[(0, 0, 16, 16)],
        blur_style=BlurStyle(method="fill", color=(7, 11, 13)),
    )
    frame = rec._grab.capture(bbox=(0, 0, 64, 48))
    assert (frame[0:16, 0:16, 0] == 7).all()
    assert (frame[0:16, 0:16, 1] == 11).all()
    assert (frame[0:16, 0:16, 2] == 13).all()


def test_cli_blur_all_warns_about_the_cost(monkeypatch, capsys):
    _run_cli(monkeypatch, ["--fullscreen", "-o", "x.mp4", "--blur-all"])
    assert "below the target fps" in capsys.readouterr().err


def test_cli_blur_all_with_fill_is_quiet(monkeypatch, capsys):
    _run_cli(monkeypatch, ["--fullscreen", "-o", "x.mp4", "--blur-all",
                           "--blur-method", "fill"])
    assert "below the target fps" not in capsys.readouterr().err


def test_cli_blur_regions_do_not_warn(monkeypatch, capsys):
    _run_cli(monkeypatch, ["--fullscreen", "-o", "x.mp4",
                           "--blur", "0,0,100,100"])
    assert "below the target fps" not in capsys.readouterr().err
