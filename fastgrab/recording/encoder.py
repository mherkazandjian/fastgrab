"""ffmpeg subprocess wrapper that consumes BGRA rawvideo frames on stdin."""
import os
import shutil
import subprocess


SUPPORTED_CODECS = ("mp4", "webm", "gif")

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
    """Escape user text for ffmpeg's drawtext text= field.

    ffmpeg's filter parser splits on ``:`` and treats ``\\`` and ``'``
    specially. We rewrite the four characters that actually break the
    parse and leave the rest alone.
    """
    out = []
    for ch in text:
        if ch in ("\\", ":", "'", "%"):
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


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
            "drawtext=fontfile={font}:text='{text}':"
            "fontcolor=white:fontsize=40:"
            "box=1:boxcolor=black@0.55:boxborderw=12:"
            "x=(w-text_w)/2:y=30:"
            "enable='lt(t,{secs})'".format(
                font=font,
                text=_escape_drawtext(title),
                secs=title_seconds,
            )
        )
    if overlay_text:
        parts.append(
            "drawtext=fontfile={font}:text='{text}':"
            "fontcolor=white@0.85:fontsize=22:"
            "box=1:boxcolor=black@0.4:boxborderw=6:"
            "x=w-text_w-20:y=20".format(
                font=font,
                text=_escape_drawtext(overlay_text),
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
        "-pix_fmt", "bgra",
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
        self.fps = fps
        self.codec = codec or infer_codec(output_path)
        self.title = title
        self.overlay_text = overlay_text
        self.font_path = font_path
        self.title_seconds = title_seconds
        self.subtitles = subtitles
        self.subtitle_style = subtitle_style
        self._proc = None
        self._frame_nbytes = width * height * 4

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
        self._proc = subprocess.Popen(
            self._build_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame) -> None:
        """Write one BGRA frame to ffmpeg's stdin.

        ``frame`` must be a contiguous ``(H, W, 4)`` ``uint8`` numpy
        array matching the encoder's configured size — this is what
        :meth:`fastgrab.screenshot.Screenshot.capture` returns.
        """
        if self._proc is None:
            raise RuntimeError("encoder not started; call start() first")
        if frame.shape != (self.height, self.width, 4):
            raise ValueError(
                "frame shape {} does not match encoder {}x{}".format(
                    frame.shape, self.width, self.height
                )
            )
        try:
            self._proc.stdin.write(frame.tobytes())
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
        if rc != 0:
            raise RuntimeError(
                "ffmpeg exited with status {}: {}".format(rc, err)
            )

    def _drain_stderr(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            data = self._proc.stderr.read() or b""
        except Exception:
            return ""
        return data.decode("utf-8", errors="replace").strip()

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
