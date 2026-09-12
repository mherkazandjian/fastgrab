"""``fastgrab-record`` CLI — argparse around :class:`Recorder`."""
import argparse
import math
import os
import re
import signal
import sys
import threading

from .clicks import CLICK_PATTERNS, ClickStyle
from .recorder import Recorder
from .subtitles import Subtitle, SubtitleStyle


XBINDKEYS_SNIPPET = """\
# fastgrab — interactive recorder (region selector + config dialog).
# Add this to ~/.xbindkeysrc, then `xbindkeys -p` to reload.
"fastgrab-record --gui"
  Print

# Or pick your own combo, e.g. Super+Shift+R:
# "fastgrab-record --gui"
#   Mod4 + Shift + r
"""


def _parse_region(value: str):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "expected 4 comma-separated ints (X,Y,W,H), got {!r}".format(value)
        )
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "region values must be integers, got {!r}".format(value)
        )
    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError(
            "region origin must be non-negative, got {},{}".format(x, y)
        )
    # Codecs aligned to yuv420p drop one odd pixel per dimension, so
    # anything below 2 px rounds down to zero at encode time.
    if w < 2 or h < 2:
        raise argparse.ArgumentTypeError(
            "region width and height must be at least 2, got {}x{}".format(w, h)
        )
    return (x, y, w, h)


def _positive_int(value: str):
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected an integer, got {!r}".format(value)
        )
    if iv <= 0:
        raise argparse.ArgumentTypeError(
            "value must be positive, got {}".format(iv)
        )
    return iv


def _parse_bgr(value: str):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected 3 comma-separated ints (B,G,R), got {!r}".format(value)
        )
    try:
        b, g, r = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "colour values must be integers, got {!r}".format(value)
        )
    if not all(0 <= v <= 255 for v in (b, g, r)):
        raise argparse.ArgumentTypeError(
            "colour values must be in 0..255, got {!r}".format(value)
        )
    return (b, g, r)


_SUBTITLE_RE = re.compile(r"^(?P<start>[0-9.]+)-(?P<end>[0-9.]+):(?P<text>.*)$")


def _parse_subtitle(value: str):
    """Parse ``START-END:TEXT`` (times in seconds) into a :class:`Subtitle`."""
    m = _SUBTITLE_RE.match(value)
    if m is None:
        raise argparse.ArgumentTypeError(
            "expected START-END:TEXT (times in seconds), got {!r}".format(value)
        )
    try:
        return Subtitle(
            text=m.group("text"),
            start=float(m.group("start")),
            end=float(m.group("end")),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_parser():
    p = argparse.ArgumentParser(
        prog="fastgrab-record",
        description="Record a region of the screen to mp4/webm/gif "
                    "(Linux/X11, draft).",
    )
    p.add_argument(
        "-o", "--output",
        help="output path; format inferred from extension (.mp4, .webm, .gif). "
             "Required unless --gui or --print-xbindkeys is passed.",
    )
    region = p.add_mutually_exclusive_group()
    region.add_argument(
        "--region", type=_parse_region, metavar="X,Y,W,H",
        help="region to capture (top-left X,Y plus W,H in pixels)",
    )
    region.add_argument(
        "--fullscreen", action="store_true",
        help="capture the whole screen",
    )
    p.add_argument(
        "--fps", type=_positive_int, default=30,
        help="target frames per second (default: 30)",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="stop after N seconds; omit to record until Ctrl-C",
    )
    p.add_argument(
        "--countdown", type=float, default=0, metavar="SECONDS",
        help="wait N seconds before capture starts (default: 0). Buys time "
             "to minimise or move the control window out of shot.",
    )
    p.add_argument(
        "--backend", default="x11", choices=["x11"],
        help="capture backend (only x11 in this draft)",
    )
    p.add_argument(
        "--title", default=None,
        help="title text shown top-centre for the first 3 seconds",
    )
    p.add_argument(
        "--overlay-text", default=None,
        help="watermark text shown top-right for the whole clip",
    )
    p.add_argument(
        "--show-clicks", action="store_true",
        help="overlay a click animation on every detected mouse click "
             "(needs the [gui] extra: python-xlib)",
    )
    p.add_argument(
        "--click-style", default="ring", choices=list(CLICK_PATTERNS),
        help="click animation pattern (default: ring)",
    )
    p.add_argument(
        "--click-color", type=_parse_bgr, default=None, metavar="B,G,R",
        help="click animation colour, e.g. 255,200,0 (default: cyan)",
    )
    p.add_argument(
        "--click-lifetime", type=float, default=None, metavar="SECONDS",
        help="how long the click animation runs (default: 0.5)",
    )
    p.add_argument(
        "--show-cursor", action="store_true",
        help="stamp an emulated arrow pointer at the mouse position — the "
             "X11 capture path never includes the real cursor sprite "
             "(needs the [gui] extra: python-xlib)",
    )
    p.add_argument(
        "--subtitle", type=_parse_subtitle, action="append", default=None,
        metavar="START-END:TEXT",
        help="timed subtitle, e.g. '1.5-4.0:Hello world'; repeatable",
    )
    p.add_argument(
        "--subtitle-font", default=None, metavar="PATH",
        help="font file for subtitles (default: $FASTGRAB_FONT or DejaVu)",
    )
    p.add_argument(
        "--subtitle-fontsize", type=int, default=28, metavar="N",
        help="subtitle font size (default: 28)",
    )
    p.add_argument(
        "--subtitle-color", default="white", metavar="COLOR",
        help="subtitle font colour as an ffmpeg colour string, e.g. "
             "white, 0xRRGGBB, red@0.8 (default: white)",
    )
    p.add_argument(
        "--subtitle-box-color", default="black@0.55", metavar="COLOR",
        help="subtitle background box colour (default: black@0.55)",
    )
    p.add_argument(
        "--subtitle-position", default="bottom", choices=["top", "bottom"],
        help="where subtitles are placed (default: bottom)",
    )
    p.add_argument(
        "--gui", action="store_true",
        help="open a region selector then a config dialog before recording",
    )
    p.add_argument(
        "--print-xbindkeys", action="store_true",
        help="print an xbindkeys snippet for binding --gui to a hotkey",
    )
    return p


def _run_gui(args):
    """Resolve --gui into a fully populated args namespace.

    The selector + dialog can each return ``None`` (user cancelled) —
    in that case we exit cleanly with status 0 so a hotkey press that
    the user backs out of doesn't show up as a failure in xbindkeys.

    ``--fullscreen`` or ``--region`` short-circuit the drag selector:
    the bbox is taken from the flag and only the config dialog is shown.
    Plain ``--gui`` opens the selector first.
    """
    from .gui import select_region, show_config_dialog

    if args.fullscreen:
        from fastgrab.screenshot import Screenshot
        w, h = Screenshot(backend=args.backend).screensize
        bbox = (0, 0, w, h)
    elif args.region is not None:
        bbox = args.region
    else:
        bbox = select_region(backend=args.backend)
        if bbox is None:
            print("fastgrab: selection cancelled", file=sys.stderr)
            return None

    defaults = {
        "fps": args.fps,
        "duration": "" if args.duration is None else str(args.duration),
        "countdown": int(args.countdown),
        "title": args.title or "",
        "overlay_text": args.overlay_text or "",
        "show_clicks": args.show_clicks,
        "output": args.output,
    }
    config = show_config_dialog(bbox=bbox, defaults=defaults)
    if config is None:
        print("fastgrab: cancelled", file=sys.stderr)
        return None

    args.region = config["bbox"]
    args.output = config["output"]
    args.fps = config["fps"]
    args.duration = config["duration"]
    args.countdown = config["countdown"]
    args.title = config["title"]
    args.overlay_text = config["overlay_text"]
    args.show_clicks = config["show_clicks"]
    return args


def _stdout_progress():
    """Return an ``on_progress`` callback that prints a live status line.

    Active only when stdout is a TTY, so piped output and hotkey launches
    (no terminal) don't get sprayed with carriage returns. Throttled to
    ~4 updates/second so a 60 fps capture doesn't thrash the terminal.
    """
    if not sys.stdout.isatty():
        return None
    state = {"last": -1.0}

    def _cb(frames, elapsed):
        if elapsed - state["last"] < 0.25:
            return
        state["last"] = elapsed
        fps = (frames / elapsed) if elapsed > 0 else 0.0
        sys.stdout.write(
            "\rrecording… {} frames  {:.1f}s  {:.1f} fps".format(
                frames, elapsed, fps
            )
        )
        sys.stdout.flush()

    return _cb


def _stdout_countdown():
    """Return an ``on_countdown`` callback that prints the seconds remaining.

    TTY-gated like :func:`_stdout_progress`. The progress line overwrites
    this one once capture begins.
    """
    if not sys.stdout.isatty():
        return None

    def _cb(remaining):
        sys.stdout.write(
            "\rstarting in {}s…   ".format(int(math.ceil(remaining)))
        )
        sys.stdout.flush()

    return _cb


def _missing_output(output):
    """The local file ffmpeg should have written, if it is absent.

    Returns ``None`` when there is nothing to complain about -- either the
    file is there, or the destination is not a local file at all.

    ffmpeg accepts protocol URLs as outputs, so the configured string is
    not always a path. ``file:`` names a local path with the prefix
    stripped; it exists precisely so a filename containing a colon, or
    one starting with a dash, can be given unambiguously. Anything else
    carrying a scheme (``rtmp://``, ``tcp://``, ``pipe:``) is not on this
    filesystem and there is nothing to look for. Checking the raw string
    failed a perfectly good recording: ``-o file:out.mp4`` writes
    ``out.mp4``, and os.path.exists never found it under that name.
    """
    if output.startswith("file:"):
        output = output[len("file:"):]
    else:
        scheme = output.split(":", 1)[0]
        # A single-letter scheme is a Windows drive, not a protocol.
        if ":" in output and len(scheme) > 1 and scheme.isalpha():
            return None
    return None if os.path.exists(output) else output


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print_xbindkeys:
        sys.stdout.write(XBINDKEYS_SNIPPET)
        return 0

    if args.gui:
        args = _run_gui(args)
        if args is None:
            return 0
        # GUI flow always populates a region (or returns None above).
        bbox = args.region
    else:
        # Force an explicit capture-target choice — earlier drafts let
        # bare `fastgrab-record -o foo.mp4` default to fullscreen, which
        # made --fullscreen a no-op flag. Now exactly one of --region /
        # --fullscreen / --gui has to be picked.
        if not args.fullscreen and args.region is None:
            parser.error(
                "specify a capture target: --fullscreen, "
                "--region X,Y,W,H, or --gui"
            )
        bbox = None if args.fullscreen else args.region

    if not args.output:
        parser.error("--output is required (or pass --gui)")

    click_style = None
    if (args.click_style != "ring" or args.click_color is not None
            or args.click_lifetime is not None):
        click_style = ClickStyle(
            pattern=args.click_style,
            color=args.click_color or ClickStyle().color,
            lifetime=(args.click_lifetime
                      if args.click_lifetime is not None
                      else ClickStyle().lifetime),
        )
    subtitle_style = SubtitleStyle(
        font_path=args.subtitle_font,
        font_size=args.subtitle_fontsize,
        font_color=args.subtitle_color,
        box_color=args.subtitle_box_color,
        position=args.subtitle_position,
    )

    recorder = Recorder(
        output_path=args.output,
        bbox=bbox,
        fps=args.fps,
        backend=args.backend,
        title=args.title,
        overlay_text=args.overlay_text,
        show_clicks=args.show_clicks,
        click_style=click_style,
        show_cursor=args.show_cursor,
        subtitles=args.subtitle,
        subtitle_style=subtitle_style,
    )

    stop_event = threading.Event()

    # Ctrl-C should stop the loop cleanly so ffmpeg gets to finalise the
    # file. The default SIGINT handler raises KeyboardInterrupt mid-write,
    # which can leave a truncated container.
    def _on_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    progress_cb = None
    try:
        if args.gui:
            # Launched from a hotkey there's no terminal to print to, so
            # show a small window with the live frame count. It stays open
            # (and is not forced on top) so you can minimise it during the
            # countdown before capture starts.
            from .gui import record_with_progress
            stats = record_with_progress(
                recorder, bbox=bbox, duration=args.duration,
                stop_event=stop_event, countdown=args.countdown,
            )
        else:
            progress_cb = _stdout_progress()
            stats = recorder.record(
                duration=args.duration, stop_event=stop_event,
                on_progress=progress_cb,
                countdown=args.countdown, on_countdown=_stdout_countdown(),
            )
    except (RuntimeError, ValueError) as exc:
        if progress_cb is not None:
            sys.stdout.write("\n")
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    if progress_cb is not None:
        # Close off the in-place status line so the summary starts fresh.
        sys.stdout.write("\n")
        sys.stdout.flush()

    if stats is None:
        return 0

    if not stats.get("encoder_started", True):
        # Ctrl-C during the countdown stops the recorder before ffmpeg is
        # ever started, so there is genuinely no file. Reporting that as
        # "wrote <path>: 0 frames" sent a script off to open something
        # that was never created. Cancelling deliberately is not an
        # error, so the exit status stays 0, matching the selection- and
        # dialog-cancel paths above.
        #
        # Keyed on whether the encoder ran, not on the frame count: if
        # the loop is stopped after ffmpeg starts but before the first
        # capture, ffmpeg writes and closes an empty container quite
        # happily -- 261 bytes for mp4, 465 for webm -- so zero frames
        # and no file are different things.
        print(
            "fastgrab: cancelled before the first frame; {} was not "
            "written".format(stats["output"]),
            file=sys.stderr,
        )
        return 0

    missing = _missing_output(stats["output"])
    if missing is not None:
        # Frames went to ffmpeg and it exited cleanly, yet nothing is
        # there. Whatever the cause, saying "wrote" would be a lie of the
        # same kind, so fail rather than describe a file that is absent.
        print(
            "error: {} frames were encoded but {} does not exist".format(
                stats["frames"], missing
            ),
            file=sys.stderr,
        )
        return 1

    duplicated = stats.get("written_frames", stats["frames"]) - stats["frames"]
    print(
        "wrote {output}: {frames} frames in {elapsed:.2f}s "
        "({fps:.1f} fps achieved{dup})".format(
            output=stats["output"],
            frames=stats["frames"],
            elapsed=stats["elapsed_seconds"],
            fps=stats["achieved_fps"],
            dup=(", {} duplicated to hold {} fps".format(duplicated, args.fps)
                 if duplicated > 0 else ""),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
