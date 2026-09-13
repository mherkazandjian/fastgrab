"""ffmpeg subprocess wrapper that consumes BGRA rawvideo frames on stdin."""
import os
import shutil
import subprocess
import tempfile

import numpy


SUPPORTED_CODECS = ("mp4", "webm", "gif")

# How timed subtitles are rendered: a drawtext chain built in this module,
# or an ASS script burned in by libass. Lives here rather than in
# subtitles.py (where it belongs conceptually) because that module imports
# helpers from this one, and the import has to stay one-directional.
SUBTITLE_BACKENDS = ("drawtext", "ass")

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


def validate_subtitle_backend(backend: str) -> str:
    """Return ``backend`` if it names a subtitle renderer, else raise.

    Checked at construction so a typo fails before the display is opened
    and ffmpeg is spawned, like :func:`validate_fps`.
    """
    if backend not in SUBTITLE_BACKENDS:
        raise ValueError(
            "subtitle_backend must be one of {}, got {!r}".format(
                ", ".join(repr(b) for b in SUBTITLE_BACKENDS), backend
            )
        )
    return backend


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


def _escape_filter_value(value: str) -> str:
    r"""Escape one level of ffmpeg filtergraph syntax.

    ffmpeg unescapes a filter option value **twice** on its way in: once
    when the graph parser splits the description into filters and their
    argument blobs, and again when the argument blob is split into
    ``key=value`` pairs. Both passes run ``av_get_token``, which turns
    ``\x`` into ``x`` for any ``x`` and treats ``'`` as a quote. So a
    value containing filter syntax has to be run through this function
    twice — see :func:`fastgrab.recording.subtitles.build_ass_filter`.

    Escaping once is the trap: ffmpeg accepts the string and then opens
    the *wrong* file, because the surviving ``'`` is read as an opening
    quote and silently swallowed (``/tmp/it's.ass`` → ``/tmp/its.ass``).

    Over-escaping is harmless — ``\x`` always collapses to ``x`` — so we
    escape the union of what matters at either level: the escape and
    quote characters, the ``:`` and ``=`` that separate options, and the
    ``,``, ``;``, ``[`` and ``]`` that separate filters.
    """
    out = []
    for char in str(value):
        if char in "\\':,;[]=":
            out.append("\\")
        out.append(char)
    return "".join(out)


def _escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use as a filter option value.

    Twice, for the two unescaping passes described in
    :func:`_escape_filter_value`. Measured against a plain path, with the
    same font copied to each awkward name: escaped once, a path
    containing an apostrophe, a colon, a comma or a bracket makes ffmpeg
    reject the whole filtergraph, and one containing a backslash renders
    visibly different pixels. Escaped twice, all five reproduce the plain
    path exactly.

    This is why a font path must not go through
    :func:`_quote_drawtext_text`: that is for the quoted ``text=`` field,
    and a path is an unquoted option value with different rules.
    """
    return _escape_filter_value(_escape_filter_value(path))


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
    # A path is an option value, not text: escaped for both parse passes
    # rather than through the text quoter. See _escape_filter_path.
    font = _escape_filter_path(font)
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


# Said instead of trailing an empty string off the end of an error. With
# -loglevel error a clean run writes nothing, so silence is expected and
# worth stating rather than leaving the reader wondering whether the
# message was truncated.
_NO_STDERR = "(ffmpeg wrote nothing to stderr)"


def _local_output_path(target: str) -> "str | None":
    """The filesystem path ffmpeg will write, or ``None`` if it is not a file.

    ffmpeg accepts protocol URLs as outputs, so the configured string is
    not always a path. ``file:`` names a local path with the prefix
    stripped -- it exists precisely so a name containing a colon, or one
    starting with a dash, can be given unambiguously. Anything else
    carrying a scheme is somewhere else entirely.

    The scheme test allows the ``+`` of composed protocols such as
    ``crypto+file:``, and treats a single letter as a Windows drive
    rather than a protocol.
    """
    if target.startswith("file:"):
        return target[len("file:"):]
    head = target.split(":", 1)[0]
    if ":" in target and len(head) > 1 and all(
            ch.isalnum() or ch in "+-." for ch in head):
        return None
    return target


def _same_file(left: str, right: str) -> bool:
    """Whether two paths name the same file, existing or not."""
    try:
        return os.path.samefile(left, right)
    except OSError:
        return os.path.realpath(left) == os.path.realpath(right)


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
    each entry's start/end window. ``subtitle_backend`` picks how:
    ``"drawtext"`` (the default) builds one drawtext filter per line,
    ``"ass"`` writes an Advanced SubStation Alpha script to a temporary
    file and burns it in with libass. ``subtitle_sidecar`` is a path to
    write that ASS script to and keep — it works with either backend, and
    naming it after the video (``demo.mp4`` → ``demo.ass``) is enough for
    mpv or VLC to offer it as a toggleable track.
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: int = 30, codec: str = None,
                 title: str = None, overlay_text: str = None,
                 font_path: str = None, title_seconds: float = 3.0,
                 subtitles=None, subtitle_style=None,
                 subtitle_backend: str = "drawtext",
                 subtitle_sidecar: str = None):
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
        self.subtitle_backend = validate_subtitle_backend(subtitle_backend)
        self.subtitle_sidecar = subtitle_sidecar
        if subtitle_sidecar:
            local = _local_output_path(subtitle_sidecar)
            if local is None:
                raise ValueError(
                    "the subtitle sidecar must be a local path, got "
                    "{!r}".format(subtitle_sidecar)
                )
            target = _local_output_path(output_path)
            if target is not None and _same_file(local, target):
                raise ValueError(
                    "the subtitle sidecar {!r} is the recording's own "
                    "output file; writing the script there would destroy "
                    "it".format(subtitle_sidecar)
                )
            if not subtitles:
                raise ValueError(
                    "a subtitle sidecar was requested but no subtitles "
                    "were given, so there would be nothing to write"
                )
        self._proc = None
        # Temporary ASS script to delete on close(); a caller-supplied
        # sidecar is never recorded here, because it is theirs to keep.
        self._ass_path = None
        self._stderr_file = None
        # The script to publish as the caller's sidecar, kept from start()
        # so close() can write it only once the encode has succeeded.
        self._sidecar_document = None

    def _build_argv(self, ass_path=None):
        """The ffmpeg command line. Pure — this writes nothing.

        ``ass_path`` is the script :meth:`start` has already written, when
        the ass backend is in use. Building argv used to write it, which
        meant merely inspecting the command line truncated whatever the
        caller had named as a sidecar.
        """
        vf = _build_drawtext_filter(
            title=self.title,
            overlay_text=self.overlay_text,
            title_seconds=self.title_seconds,
            font_path=self.font_path,
        )
        sub_vf = self._build_subtitle_filter(ass_path)
        if sub_vf:
            vf = "{},{}".format(vf, sub_vf) if vf else sub_vf
        return _ffmpeg_args(
            self.codec, self.width, self.height, self.fps, self.output_path,
            vf_extra=vf,
        )

    def ass_document(self):
        """The ASS script this configuration needs, or ``None``.

        Built whenever the ``ass`` backend is selected *or* a sidecar was
        asked for, so a sidecar can be exported while drawtext does the
        burning in.
        """
        if not self.subtitles:
            return None
        if self.subtitle_backend != "ass" and not self.subtitle_sidecar:
            return None
        from .subtitles import build_ass_document
        return build_ass_document(
            self.subtitles, self.subtitle_style,
            width=self.width, height=self.height,
        )

    def _build_subtitle_filter(self, ass_path=None):
        """The ``-vf`` fragment for subtitles, or ``None``."""
        if not self.subtitles:
            return None
        from .subtitles import (
            ass_fonts_dir, build_ass_filter, build_subtitle_filters,
        )
        if self.subtitle_backend != "ass":
            return build_subtitle_filters(self.subtitles, self.subtitle_style)
        if ass_path is None:
            raise RuntimeError(
                "the ass backend needs its script written before the "
                "command line is built"
            )
        return build_ass_filter(
            ass_path, fontsdir=ass_fonts_dir(self.subtitle_style)
        )

    def _write_ass(self, document: str) -> str:
        """Write ``document`` to a private temp file and return its path.

        Always private, never the caller's sidecar. Two reasons, and the
        second is not obvious:

        * pointing ffmpeg at the sidecar meant this write truncated it,
          and it happened while *building* the command line -- before any
          capture, and with the default drawtext backend, so a mistyped
          --subtitle-sidecar destroyed the named file for a recording
          that never used ASS at all;
        * ffmpeg opens the script lazily. Deleting it after Popen returns
          but before the first frame still fails ASS initialisation, so a
          successful Popen does not mean the script has been read. Two
          recordings sharing a sidecar path could each consume the
          other's.

        The caller's sidecar is published from this same document once
        the encode has succeeded -- see :meth:`_publish_sidecar`.
        """
        self._cleanup_ass()
        handle, path = tempfile.mkstemp(prefix="fastgrab-", suffix=".ass")
        with os.fdopen(handle, "w", encoding="utf-8") as fobj:
            fobj.write(document)
        self._ass_path = path
        return path

    def _publish_sidecar(self, document: str) -> None:
        """Write the caller's sidecar, atomically, once encoding worked.

        Staged through a temp file in the same directory and renamed, so
        a failure part-way cannot leave the destination truncated: either
        the old contents survive or the new ones are complete.
        """
        destination = _local_output_path(self.subtitle_sidecar)

        # Checked again here, not only at construction. Back then neither
        # file existed, so samefile() could not answer and the fallback
        # compared realpath strings -- which on a case-insensitive
        # filesystem says clip.mp4 and CLIP.MP4 are different. By now
        # ffmpeg has created the video, so the question can be answered
        # properly, and getting it wrong would replace the recording that
        # was just made with a few hundred bytes of subtitle script.
        target = _local_output_path(self.output_path)
        if target is not None and _same_file(destination, target):
            raise RuntimeError(
                "refusing to write the subtitle sidecar {!r}: it is the "
                "recording that was just written to {!r}".format(
                    self.subtitle_sidecar, self.output_path
                )
            )

        folder = os.path.dirname(os.path.abspath(destination)) or "."
        handle = None
        staged = None
        try:
            handle, staged = tempfile.mkstemp(
                prefix=".fastgrab-", suffix=".ass", dir=folder
            )
            with os.fdopen(handle, "w", encoding="utf-8") as fobj:
                handle = None
                fobj.write(document)
            os.replace(staged, destination)
            staged = None
        except OSError as exc:
            if handle is not None:
                os.close(handle)
            if staged is not None:
                try:
                    os.remove(staged)
                except OSError:
                    pass
            raise RuntimeError(
                "cannot write the subtitle sidecar {!r}: {}".format(
                    self.subtitle_sidecar, exc
                )
            ) from exc

    def _cleanup_ass(self) -> None:
        """Delete the temporary ASS script, if we wrote one."""
        path, self._ass_path = self._ass_path, None
        if path is None:
            return
        try:
            os.remove(path)
        except OSError:
            pass

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
        # no timeout on either side. A file has no capacity to reach.
        document = self.ass_document()
        ass_path = self._write_ass(document) if document is not None else None
        self._sidecar_document = document

        self._stderr_file = tempfile.TemporaryFile()
        try:
            argv = self._build_argv(ass_path)
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
            )
        except Exception:
            # close() never runs when start() fails, so both of these have
            # to be released here: the stderr file, or a caller that
            # retries leaks one per attempt, and the temporary ASS script,
            # or it is left behind in the system temp directory.
            self._stderr_file.close()
            self._stderr_file = None
            self._cleanup_ass()
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
            self._cleanup_ass()
            return
        try:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except BrokenPipeError:
                # ffmpeg has already gone. Letting this out would report
                # the broken pipe -- a symptom that names nothing -- and
                # skip the exit status and stderr below, which say why it
                # went. Fall through and let those do the talking.
                pass
            try:
                rc = self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
                # Drain before the finally closes the file. A hang is the
                # case where ffmpeg's own words matter most and the one
                # where they used to be dropped: the message named only
                # the timeout, while the finally computed the stderr and
                # then discarded it.
                raise RuntimeError(
                    "ffmpeg did not exit within {}s and was killed: {}".format(
                        timeout, self._drain_stderr() or _NO_STDERR
                    )
                )
        finally:
            err = self._drain_stderr()
            self._proc = None
            # ffmpeg has exited, so libass is done with the script.
            self._cleanup_ass()
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None
        if rc != 0:
            raise RuntimeError(
                "ffmpeg exited with status {}: {}".format(rc, err or _NO_STDERR)
            )

        # Only now. A sidecar published before the encode ran would
        # describe a recording that does not exist, and would already
        # have replaced whatever was at that path.
        document, self._sidecar_document = self._sidecar_document, None
        if self.subtitle_sidecar and document is not None:
            self._publish_sidecar(document)

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
