"""``fastgrab-record`` CLI — argparse around :class:`Recorder`."""
import argparse
import math
import re
import signal
import sys
import threading

from fastgrab.effects import (
    BLUR_METHODS,
    DEFAULT_BLOCK,
    DEFAULT_IMAGE_FIT,
    DEFAULT_RADIUS,
    IMAGE_FITS,
    PIXELATE_METHODS,
    BlurStyle,
)

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


def _parse_xywh(value: str, min_size: int):
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
    if w < min_size or h < min_size:
        raise argparse.ArgumentTypeError(
            "region width and height must be at least {}, got {}x{}".format(
                min_size, w, h
            )
        )
    return (x, y, w, h)


def _parse_region(value: str):
    # Codecs aligned to yuv420p drop one odd pixel per dimension, so a
    # capture region below 2 px rounds down to zero at encode time.
    return _parse_xywh(value, 2)


def _parse_blur_region(value: str):
    # Blur regions are clipped to the frame rather than encoded, so a
    # single-pixel rectangle is meaningless but harmless.
    return _parse_xywh(value, 1)


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


def _load_cover_image(parser, path):
    """Decode --blur-image into a BGR numpy array.

    Pillow is lazy and optional, exactly as python-xlib is for the click
    overlays: the core blur takes an array and needs no decoder at all,
    so only this convenience pays for one.
    """
    try:
        from PIL import Image
    except ImportError:
        parser.error(
            "--blur-image needs Pillow to decode {}: "
            "pip install fastgrab[gui]".format(path)
        )
    import numpy

    try:
        with Image.open(path) as handle:
            rgb = numpy.asarray(handle.convert("RGB"), dtype=numpy.uint8)
    except OSError as exc:
        parser.error("--blur-image could not read {}: {}".format(path, exc))
    # PIL gives RGB; frames are BGR.
    return numpy.ascontiguousarray(rgb[..., ::-1])


def _nonnegative_int(value: str):
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected an integer, got {!r}".format(value)
        )
    if iv < 0:
        raise argparse.ArgumentTypeError(
            "value must not be negative, got {}".format(iv)
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
    blur = p.add_mutually_exclusive_group()
    blur.add_argument(
        "--blur", type=_parse_blur_region, action="append", default=None,
        metavar="X,Y,W,H",
        help="obscure this screen region in every frame; repeatable",
    )
    blur.add_argument(
        "--blur-all", action="store_true",
        help="obscure the whole captured frame",
    )
    p.add_argument(
        "--blur-method", default="box", choices=list(BLUR_METHODS),
        help="how blurred regions are obscured (default: box). fill and "
             "pixelate-random destroy the pixels outright; "
             "pixelate-random-shuffle keeps the real tile colours but "
             "scrambles their positions; box, gaussian and pixelate are "
             "cosmetic. Use fill or pixelate-random for secrets.",
    )
    p.add_argument(
        "--blur-radius", type=_positive_int, default=None, metavar="N",
        help="blur kernel radius in pixels for box/gaussian "
             "(default: {})".format(DEFAULT_RADIUS),
    )
    p.add_argument(
        "--blur-block", type=_positive_int, default=None, metavar="N",
        help="mosaic tile size in pixels for the pixelate methods "
             "(default: {})".format(DEFAULT_BLOCK),
    )
    p.add_argument(
        "--blur-seed", type=_nonnegative_int, default=None, metavar="N",
        help="seed for the pixelate-random methods (default: 0). Fixed "
             "rather than per-frame on purpose: re-rolling every frame "
             "would let a recording be averaged back towards what is "
             "underneath.",
    )
    p.add_argument(
        "--blur-image", default=None, metavar="PATH",
        help="image to stamp over the region for --blur-method image. "
             "Stretched to fit. Decoding needs Pillow "
             "(pip install fastgrab[gui]); the Python API takes a numpy "
             "array and needs nothing extra.",
    )
    p.add_argument(
        "--blur-image-fit", default=None, choices=list(IMAGE_FITS),
        help="how the cover image is mapped onto the region (default: {}). "
             "crop scales it to cover and trims the overflow, fit scales it "
             "to sit inside and pads with --blur-color, stretch distorts it "
             "to the exact shape, tile repeats it at its own size. Only "
             "stretch changes the picture's proportions.".format(
                 DEFAULT_IMAGE_FIT),
    )
    p.add_argument(
        "--blur-color", type=_parse_bgr, default=None, metavar="B,G,R",
        help="colour for --blur-method fill, e.g. 255,255,255 "
             "(default: 0,0,0, a black box)",
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
    blur = True if args.blur_all else args.blur
    blur_style = None
    blur_tuned = (
        args.blur_method != "box" or args.blur_radius is not None
        or args.blur_block is not None or args.blur_color is not None
        or args.blur_seed is not None or args.blur_image is not None
        or args.blur_image_fit is not None
    )
    if blur_tuned and not blur:
        # Silently ignoring a --blur-method the user typed would hide a
        # failed redaction, which is the one mistake that actually leaks.
        parser.error(
            "--blur-method/--blur-radius/--blur-block/--blur-color need a "
            "target: pass --blur X,Y,W,H or --blur-all"
        )
    if blur_tuned:
        # Each tuning flag belongs to specific methods. Accepting
        # --blur-color with a box blur would leave the user believing
        # they had painted a solid box over a secret when they had only
        # softened it, so refuse rather than ignore.
        applies_to = {
            "--blur-radius": ("box", "gaussian"),
            "--blur-block": PIXELATE_METHODS,
            # padding colour for --blur-image-fit fit, as well as the
            # block colour for fill
            "--blur-color": ("fill", "image"),
            "--blur-seed": ("pixelate-random", "pixelate-random-shuffle"),
            "--blur-image": ("image",),
            "--blur-image-fit": ("image",),
        }
        given = [
            name for name, value in (
                ("--blur-radius", args.blur_radius),
                ("--blur-block", args.blur_block),
                ("--blur-color", args.blur_color),
                ("--blur-seed", args.blur_seed),
                ("--blur-image", args.blur_image),
                ("--blur-image-fit", args.blur_image_fit),
            ) if value is not None
        ]
        ignored = [
            name for name in given
            if args.blur_method not in applies_to[name]
        ]
        if ignored:
            parser.error(
                "{} {} nothing for --blur-method {} (it applies to {}); "
                "a redaction option that is silently ignored is worse than "
                "an error".format(
                    " and ".join(ignored),
                    "does" if len(ignored) == 1 else "do",
                    args.blur_method,
                    ", ".join(
                        sorted({m for n in ignored for m in applies_to[n]})
                    ),
                )
            )
    if blur_tuned:
        cover = None
        if args.blur_image is not None:
            cover = _load_cover_image(parser, args.blur_image)
        try:
            blur_style = BlurStyle(
                method=args.blur_method,
                radius=(args.blur_radius if args.blur_radius is not None
                        else BlurStyle().radius),
                block=(args.blur_block if args.blur_block is not None
                       else BlurStyle().block),
                seed=(args.blur_seed if args.blur_seed is not None
                      else BlurStyle().seed),
                color=args.blur_color or BlurStyle().color,
                image=cover,
                image_fit=(args.blur_image_fit if args.blur_image_fit
                           is not None else BlurStyle().image_fit),
            )
        except ValueError as exc:
            # BlurStyle rejects identity settings such as
            # --blur-method pixelate --blur-block 1; surface that as a
            # usage error instead of a traceback.
            parser.error(str(exc))

    if blur is True and (blur_style is None
                         or blur_style.method in ("box", "gaussian")):
        # Measured in the dev container: a full 1080p frame costs ~64 ms
        # (box) / ~180 ms (gaussian) per frame, so capture cannot hold
        # 30 fps. The recorder duplicates frames to keep the clip's
        # duration honest, so the output is still correct — just choppy.
        # Say so rather than letting the fps quietly collapse.
        print(
            "note: --blur-all with --blur-method {} costs tens of "
            "milliseconds per frame at 1080p and will hold capture below "
            "the target fps; pixelate and fill are much cheaper".format(
                "box" if blur_style is None else blur_style.method
            ),
            file=sys.stderr,
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
        blur=blur,
        blur_style=blur_style,
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
