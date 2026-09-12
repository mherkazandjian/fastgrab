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
import sys
import tempfile
import threading
import time

import numpy
import pytest

from fastgrab.recording import (
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
    enc._build_argv = lambda: [
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


def _stats(output, frames=0, written=0):
    return {
        "frames": frames,
        "written_frames": written,
        "elapsed_seconds": 0.0 if not frames else 1.0,
        "achieved_fps": 0.0 if not frames else float(frames),
        "output": str(output),
    }


def test_a_cancelled_recording_does_not_claim_a_file(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out.mp4"
    rc = _run_cli_with(monkeypatch, _stats(out), tmp_path)
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
