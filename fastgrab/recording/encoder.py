"""ffmpeg subprocess wrapper that consumes BGRA rawvideo frames on stdin."""
import os
import shutil
import subprocess
import tempfile

import numpy


SUPPORTED_CODECS = ("mp4", "webm", "gif")

# Codecs that encode to yuv420p and therefore need even width/height.
# gif goes through a palette filter and has no such constraint.
_EVEN_DIMENSION_CODECS = ("mp4", "webm")


def validate_fps(fps) -> int:
    """Return ``fps`` as an int, or raise :class:`ValueError`.

    Shared by :class:`FfmpegEncoder` and
    :class:`fastgrab.recording.Recorder` so API callers get a clear error
    instead of a ``ZeroDivisionError`` from the capture loop or a cryptic
    ffmpeg failure.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("fps must be a positive number, got {!r}".format(fps))
    if fps <= 0 or fps != int(fps):
        raise ValueError(
            "fps must be a positive integer, got {!r}".format(fps)
        )
    return int(fps)


def validate_dimensions(codec: str, width: int, height: int) -> None:
    """Raise :class:`ValueError` if ``width``/``height`` can't be encoded.

    mp4 (libx264) and webm (libvpx-vp9) are written as yuv420p, which
    requires even dimensions; ffmpeg otherwise fails late with a generic
    error. :class:`Recorder` rounds its region down to satisfy this, but
    a standalone :class:`FfmpegEncoder` gets whatever the caller passes.
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            "width and height must be positive, got {}x{}".format(
                width, height
            )
        )
    if codec in _EVEN_DIMENSION_CODECS and (width % 2 or height % 2):
        raise ValueError(
            "{} output requires even width and height (yuv420p), got "
            "{}x{}".format(codec, width, height)
        )

# DejaVu ships on Debian/Ubuntu (and the dev container) by default, so
# probing this short list gets us a working drawtext filter without
# making the user pick a font path. Override with ``$FASTGRAB_FONT``.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _find_font():
    override = os.environ.get("FASTGRAB_FONT")
    if override and os.path.exists(override):
        return override
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def infer_codec(path: str) -> str:
    """Map an output filename to one of :data:`SUPPORTED_CODECS`.

    Raises :class:`ValueError` for unrecognised extensions so callers
    fail loudly instead of silently producing the wrong container.
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in SUPPORTED_CODECS:
        return ext
    raise ValueError(
        "unsupported output extension {!r}; expected one of {}".format(
            ext, ", ".join("." + c for c in SUPPORTED_CODECS)
        )
    )


def _escape_drawtext(text: str) -> str:
    """Escape a value for an *unquoted* drawtext option — a path, say.

    ffmpeg's filter parser splits options on ``:`` and treats ``\\`` and
    ``'`` specially, so those are backslash-escaped here. Use
    :func:`_quote_drawtext_text` for anything going into ``text=``, which
    is quoted and follows different rules.
    """
    out = []
    for ch in text:
        if ch in ("\\", ":", "'", "%"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _quote_drawtext_text(text: str) -> str:
    """Render ``text`` as a complete, quoted drawtext ``text=`` value.

    An apostrophe cannot simply be backslash-escaped here. A filter
    option inside a filtergraph is unescaped *twice* — once when the
    graph is split into filters, once when a filter's arguments are
    split — and a single-quoted section has no escape mechanism at all;
    it ends at the next quote. ``\\'`` therefore does not survive: it is
    consumed on the way in and the apostrophe is silently dropped, so
    ``--title "Don't stop"`` rendered "Dont stop", and with the trailing
    ``enable=`` option present the stray quote ran the parse off the end
    and ffmpeg rejected the whole filtergraph with "Filter not found" —
    no recording at all.

    What works is to close the quote, emit the apostrophe escaped for
    both levels, and reopen: ``'\\\''``. That was not deduced from the
    documentation but measured — every candidate was rendered and
    compared pixel-for-pixel against the same text supplied through
    ``textfile=``, which takes its content verbatim and so is ground
    truth. Of the encodings tried it is the only one that reproduces the
    reference.

    The single backslash used for ``:``, ``\\`` and ``%`` was checked the
    same way and does reproduce the reference, so it stays. Commas,
    brackets and semicolons need nothing: the surrounding quotes already
    protect them from the filtergraph splitter.
    """
    # Spelled out rather than written as one literal: the
    # sequence is quote, three backslashes, quote, quote, and a
    # nested escape of that is very easy to miscount.
    apostrophe = "'" + "\\" * 3 + "''"
    out = []
    for ch in text:
        if ch == "'":
            out.append(apostrophe)
        elif ch in ("\\", ":", "%"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "'" + "".join(out) + "'"


def _build_drawtext_filter(title: str = None, overlay_text: str = None,
                           title_seconds: float = 3.0,
                           font_path: str = None) -> str:
    """Compose the ``-vf`` filter string for title + overlay text.

    Returns ``None`` if neither is set or if no usable font was found.
    The title is centred at the top and shown only for the first
    ``title_seconds`` (drawtext's ``enable=lt(t,N)``); the overlay text
    sits in the top-right and is shown for the whole clip.
    """
    if not title and not overlay_text:
        return None
    font = font_path or _find_font()
    if font is None:
        return None
    # The font path goes into the same filter string as the text, so it
    # needs the same escaping — an unescaped ':' (Windows drive letters)
    # would be parsed as a filter-option separator.
    font = _escape_drawtext(font)
    parts = []
    if title:
        parts.append(
            "drawtext=fontfile={font}:text={text}:expansion=none:"
            "fontcolor=white:fontsize=40:"
            "box=1:boxcolor=black@0.55:boxborderw=12:"
            "x=(w-text_w)/2:y=30:"
            "enable='lt(t,{secs})'".format(
                font=font,
                text=_quote_drawtext_text(title),
                secs=title_seconds,
            )
        )
    if overlay_text:
        parts.append(
            "drawtext=fontfile={font}:text={text}:expansion=none:"
            "fontcolor=white@0.85:fontsize=22:"
            "box=1:boxcolor=black@0.4:boxborderw=6:"
            "x=w-text_w-20:y=20".format(
                font=font,
                text=_quote_drawtext_text(overlay_text),
            )
        )
    return ",".join(parts)


def _ffmpeg_args(codec: str, width: int, height: int, fps: int, output: str,
                 vf_extra: str = None):
    """Build the argv for the encoder subprocess.

    The input side is identical for every codec — rawvideo BGRA at
    ``width×height@fps``. Only the output side (codec + flags) varies.
    ``vf_extra`` is prepended to any codec-specific filter chain.
    """
    common_in = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        # bgr0, not bgra: the fourth byte of a fastgrab frame is unused
        # padding, not transparency (X11's XGetImage leaves it zero on a
        # 24-bit visual). Describing it as bgra told ffmpeg every pixel
        # was fully transparent, which mp4/webm ignore -- they force
        # yuv420p -- but GIF does not: paletteuse treats alpha below its
        # default threshold of 128 as transparent, so every recorded GIF
        # came out completely invisible.
        "-pix_fmt", "bgr0",
        "-s", "{}x{}".format(width, height),
        "-r", str(fps),
        "-i", "-",
    ]
    if codec == "mp4":
        out = [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
        ]
        if vf_extra:
            out += ["-vf", vf_extra]
        out.append(output)
    elif codec == "webm":
        out = [
            "-c:v", "libvpx-vp9",
            "-b:v", "2M",
            "-pix_fmt", "yuv420p",
        ]
        if vf_extra:
            out += ["-vf", vf_extra]
        out.append(output)
    elif codec == "gif":
        # Single-pass palette filter — good enough for a draft. A
        # two-pass palettegen/paletteuse split produces sharper colours
        # but doubles the wall time and isn't worth the complexity yet.
        gif_chain = (
            "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        )
        # If drawtext is requested, run it before the palette split so
        # the overlays get baked into the same colour quantisation.
        full = "{},{}".format(vf_extra, gif_chain) if vf_extra else gif_chain
        out = ["-vf", full, "-loop", "0", output]
    else:
        raise ValueError("unknown codec {!r}".format(codec))
    return common_in + out


class FfmpegEncoder:
    """Spawn ffmpeg, feed it BGRA frames, and finalise on close.

    Use as a context manager::

        with FfmpegEncoder("out.mp4", w, h, fps=30) as enc:
            for frame in frames:
                enc.write_frame(frame)

    ``title`` is rendered top-centre for the first 3 seconds; ``overlay_text``
    is a persistent watermark in the top-right. Both are optional and
    silently skipped when no usable font is found on the system.

    ``subtitles`` is a list of :class:`fastgrab.recording.subtitles.Subtitle`
    rendered (bottom-centre by default, see ``subtitle_style``) during
    each entry's start/end window via additional drawtext filters.
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: int = 30, codec: str = None,
                 title: str = None, overlay_text: str = None,
                 font_path: str = None, title_seconds: float = 3.0,
                 subtitles=None, subtitle_style=None):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = validate_fps(fps)
        self.codec = codec or infer_codec(output_path)
        validate_dimensions(self.codec, width, height)
        self.title = title
        self.overlay_text = overlay_text
        self.font_path = font_path
        self.title_seconds = title_seconds
        self.subtitles = subtitles
        self.subtitle_style = subtitle_style
        self._proc = None
        self._stderr_file = None

    def _build_argv(self):
        vf = _build_drawtext_filter(
            title=self.title,
            overlay_text=self.overlay_text,
            title_seconds=self.title_seconds,
            font_path=self.font_path,
        )
        if self.subtitles:
            # Imported here to keep the module cycle (subtitles reuses
            # helpers from this module) one-directional at import time.
            from .subtitles import build_subtitle_filters
            sub_vf = build_subtitle_filters(self.subtitles,
                                            self.subtitle_style)
            if sub_vf:
                vf = "{},{}".format(vf, sub_vf) if vf else sub_vf
        return _ffmpeg_args(
            self.codec, self.width, self.height, self.fps, self.output_path,
            vf_extra=vf,
        )

    def start(self):
        if self._proc is not None:
            raise RuntimeError("encoder already started")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH — install it (e.g. "
                "'apt-get install ffmpeg' or 'brew install ffmpeg') "
                "to use fastgrab.recording"
            )
        # stderr goes to a temp file, not a pipe. A pipe holds 64 KiB
        # (F_GETPIPE_SZ on Linux) and nothing reads this one until
        # close(), so an ffmpeg that filled it would block writing its
        # own diagnostics -- and an ffmpeg blocked on stderr stops
        # reading stdin, which blocks write_frame(), which is a hang with
        # no timeout on either side. Verified with this exact Popen
        # shape: a child writing 16 KiB to stderr completes, one writing
        # 128 KiB deadlocks the writer indefinitely.
        #
        # -loglevel error keeps real ffmpeg silent today -- an mp4, webm
        # and gif encode of 120 frames each produced 0 bytes, and so did
        # titles with glyphs the font lacks -- so this is a latent hazard
        # rather than one reachable now. It is worth removing anyway: the
        # safety rests entirely on a log level no test enforces, and the
        # failure it guards is an unkillable hang rather than an error.
        # A file has no capacity limit to reach.
        self._stderr_file = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                self._build_argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
            )
        except Exception:
            # close() never runs when start() fails, so the file has to be
            # released here or a caller that retries leaks one per attempt.
            self._stderr_file.close()
            self._stderr_file = None
            raise

    def write_frame(self, frame) -> None:
        """Write one BGRA frame to ffmpeg's stdin.

        ``frame`` must be a ``(H, W, 4)`` ``uint8`` numpy array matching
        the encoder's configured size — this is what
        :meth:`fastgrab.screenshot.Screenshot.capture` returns. The
        array's buffer is handed to the pipe directly (no ``tobytes()``
        copy); a non-contiguous view is made contiguous first.
        """
        if self._proc is None:
            raise RuntimeError("encoder not started; call start() first")
        if frame.shape != (self.height, self.width, 4):
            raise ValueError(
                "frame shape {} does not match encoder {}x{}".format(
                    frame.shape, self.width, self.height
                )
            )
        if frame.dtype != numpy.uint8:
            raise ValueError(
                "frame dtype must be uint8, got {}".format(frame.dtype)
            )
        if not frame.flags.c_contiguous:
            frame = numpy.ascontiguousarray(frame)
        try:
            self._proc.stdin.write(frame)
        except BrokenPipeError as exc:
            err = self._drain_stderr()
            raise RuntimeError(
                "ffmpeg closed its stdin unexpectedly: " + err
            ) from exc

    def close(self, timeout: float = 30.0) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            rc = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            raise RuntimeError("ffmpeg did not exit within {}s".format(timeout))
        finally:
            err = self._drain_stderr()
            self._proc = None
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None
        if rc != 0:
            raise RuntimeError(
                "ffmpeg exited with status {}: {}".format(rc, err)
            )

    def _drain_stderr(self, limit: int = 64 * 1024) -> str:
        """Read what ffmpeg has written to stderr so far.

        Reads the temp file rather than a pipe, so this never blocks and
        can be called while ffmpeg is still running — which write_frame()
        does, on a pipe that has just broken.

        Positional reads, not seek()+read(): subprocess hands the child a
        dup of this file's descriptor, and a dup shares the file
        *description*, offset included. Seeking here would move where
        ffmpeg's next write lands and scribble over its own log.

        Only the tail is returned. The file is unbounded by design, and a
        run that produced megabytes of diagnostics would otherwise put
        all of it into an exception message; the last lines are the ones
        that say why ffmpeg stopped.
        """
        handle = self._stderr_file
        if handle is None:
            return ""
        try:
            fd = handle.fileno()
            size = os.fstat(fd).st_size
            start = max(0, size - limit)
            if hasattr(os, "pread"):
                data = os.pread(fd, size - start, start) or b""
            else:
                # No pread off Unix. Recording is X11-only, so this is a
                # fallback for completeness rather than a supported path;
                # the offset caveat above applies to it.
                handle.seek(start)
                data = handle.read() or b""
        except Exception:
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        if start:
            text = "[...{} earlier bytes omitted...] {}".format(start, text)
        return text

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                pass
        return False
