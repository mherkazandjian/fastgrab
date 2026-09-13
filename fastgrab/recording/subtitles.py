"""Timed subtitles, rendered by ffmpeg as drawtext filters or as ASS.

A :class:`Subtitle` carries text plus a start/end window in seconds from
the start of the recording. Two backends turn a list of them into
something ffmpeg understands, selected with ``subtitle_backend``:

``"drawtext"`` (the default)
    :func:`build_subtitle_filters` returns a comma-joined chain of
    ``drawtext`` filters (one per subtitle, gated with
    ``enable='between(t,start,end)'``) that
    :class:`fastgrab.recording.encoder.FfmpegEncoder` appends to its
    ``-vf`` argument.

``"ass"``
    :func:`build_ass_document` renders the same list as an Advanced
    SubStation Alpha script. The encoder writes it to a file and burns it
    in with ffmpeg's libass-backed ``ass`` filter
    (:func:`build_ass_filter`). ASS is a real subtitle format, so the
    script can also be kept as an editable sidecar next to the video
    (``subtitle_sidecar=``), where players such as mpv and VLC pick it up
    automatically. Note it is an editable copy, not a selectable
    track: the subtitles are burned into the video either way, so a
    player that loads the sidecar as well shows the text twice.

Both paths are pure string generation — the rendering is done entirely by
ffmpeg, so there are no new Python runtime dependencies.

Colours are written as ffmpeg colour strings (``white``, ``0xRRGGBB``,
``name@alpha``) in both backends. drawtext takes them verbatim; the ASS
backend has to parse them into ``&HAABBGGRR`` (see :func:`_ass_color`),
which is why it understands a common subset of ffmpeg's colour names
rather than all ~150 of them.
"""
import os
from dataclasses import dataclass

from .encoder import (
    _escape_drawtext,
    _escape_filter_path,
    _escape_filter_value,
    _find_font,
    _quote_drawtext_text,
)


@dataclass
class Subtitle:
    """One subtitle line shown from ``start`` to ``end`` seconds."""

    text: str
    start: float
    end: float

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(
                "subtitle end ({}) must be after start ({})".format(
                    self.end, self.start
                )
            )


@dataclass
class SubtitleStyle:
    """Font, colour, and placement for subtitles.

    ``font_path`` falls back to the encoder's font discovery
    (``$FASTGRAB_FONT`` then the DejaVu candidates) when ``None``.
    ``font_color`` / ``box_color`` are ffmpeg colour strings —
    ``white``, ``0xRRGGBB``, ``red@0.8`` all work. ``position`` is
    ``"bottom"`` (centred, 40 px above the lower edge) or ``"top"``
    (centred, 30 px below the upper edge).

    ``font_name`` is used by the ``ass`` backend only: libass resolves
    fonts by family name through fontconfig, not by path, so there is
    nowhere to put ``font_path``. When it is ``None`` the family name is
    guessed from the font file's stem (``DejaVuSans-Bold.ttf`` →
    ``"DejaVuSans-Bold"``), and the font file's directory is handed to
    libass as ``fontsdir`` so a font outside the system font paths is
    still found. Set it explicitly when the guess does not match the
    family name recorded inside the font.
    """

    font_path: str = None
    font_size: int = 28
    font_color: str = "white"
    box_color: str = "black@0.55"
    border: int = 8
    position: str = "bottom"
    font_name: str = None

    def __post_init__(self):
        if self.position not in ("top", "bottom"):
            raise ValueError(
                "subtitle position must be 'top' or 'bottom', got "
                "{!r}".format(self.position)
            )


def build_subtitle_filters(subtitles, style=None) -> "str | None":
    """Compose the drawtext filter chain for ``subtitles``.

    Returns ``None`` when there is nothing to render or no usable font
    is found — the same graceful-skip semantics as the title/overlay
    filters in the encoder.
    """
    if not subtitles:
        return None
    style = style or SubtitleStyle()
    font = style.font_path or _find_font()
    if font is None:
        return None
    # A path is an option value, not text: escaped for both of ffmpeg's
    # parse passes. See _escape_filter_path.
    font = _escape_filter_path(font)
    if style.position == "bottom":
        xy = "x=(w-text_w)/2:y=h-text_h-40"
    else:
        xy = "x=(w-text_w)/2:y=30"
    parts = []
    for sub in subtitles:
        parts.append(
            "drawtext=fontfile={font}:text={text}:expansion=none:"
            "fontcolor={color}:fontsize={size}:"
            "box=1:boxcolor={box}:boxborderw={border}:"
            "{xy}:enable='between(t,{start},{end})'".format(
                font=font,
                text=_quote_drawtext_text(sub.text),
                color=style.font_color,
                size=style.font_size,
                box=style.box_color,
                border=style.border,
                xy=xy,
                start=sub.start,
                end=sub.end,
            )
        )
    return ",".join(parts)


# --------------------------------------------------------------------------
# ASS (Advanced SubStation Alpha)
# --------------------------------------------------------------------------

# The two Format: lines an ASS script declares before its Style: and
# Dialogue: rows. The field order is not fixed by the format — it is
# whatever the Format line says — but sticking to the canonical order
# keeps the output readable in Aegisub and diffable against other tools.
_ASS_STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)
_ASS_EVENT_FORMAT = (
    "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)

# Every event references this one style; the name is arbitrary but
# "Default" is what players and editors expect to find.
_ASS_STYLE_NAME = "Default"

# numpad-style alignment: 2 = bottom centre, 8 = top centre.
_ASS_ALIGNMENT = {"bottom": 2, "top": 8}
# Vertical margins chosen to land where the drawtext chain puts the text:
# y=h-text_h-40 for bottom, y=30 for top.
_ASS_MARGIN_V = {"bottom": 40, "top": 30}
# Horizontal margins have no drawtext equivalent — drawtext centres the
# line and lets a long one run off the edge, libass wraps inside them.
_ASS_MARGIN_H = 10

# Subset of ffmpeg's colour-name table (libavutil/parseutils.c), which
# follows CSS naming — note "green" is 0x008000 and "lime" is 0x00FF00.
# Only the ASS backend needs this, because it has to convert to
# &HAABBGGRR; drawtext gets the name passed through untouched.
_COLOR_NAMES = {
    "aqua": (0x00, 0xFF, 0xFF),
    "black": (0x00, 0x00, 0x00),
    "blue": (0x00, 0x00, 0xFF),
    "brown": (0xA5, 0x2A, 0x2A),
    "cyan": (0x00, 0xFF, 0xFF),
    "darkgray": (0xA9, 0xA9, 0xA9),
    "darkgrey": (0xA9, 0xA9, 0xA9),
    "fuchsia": (0xFF, 0x00, 0xFF),
    "gold": (0xFF, 0xD7, 0x00),
    "gray": (0x80, 0x80, 0x80),
    "green": (0x00, 0x80, 0x00),
    "grey": (0x80, 0x80, 0x80),
    "lightgray": (0xD3, 0xD3, 0xD3),
    "lightgrey": (0xD3, 0xD3, 0xD3),
    "lime": (0x00, 0xFF, 0x00),
    "magenta": (0xFF, 0x00, 0xFF),
    "maroon": (0x80, 0x00, 0x00),
    "navy": (0x00, 0x00, 0x80),
    "olive": (0x80, 0x80, 0x00),
    "orange": (0xFF, 0xA5, 0x00),
    "orangered": (0xFF, 0x45, 0x00),
    "pink": (0xFF, 0xC0, 0xCB),
    "purple": (0x80, 0x00, 0x80),
    "red": (0xFF, 0x00, 0x00),
    "silver": (0xC0, 0xC0, 0xC0),
    "teal": (0x00, 0x80, 0x80),
    "white": (0xFF, 0xFF, 0xFF),
    "yellow": (0xFF, 0xFF, 0x00),
}


def _ass_time(seconds) -> str:
    """Format ``seconds`` as an ASS ``H:MM:SS.cc`` timecode.

    ASS resolution is a centisecond, so times are rounded to 1/100 s.
    Hours are not zero-padded (``0:00:01.50``) and grow past one digit
    for very long recordings, which libass and every player accept.
    Negative values clamp to zero — :class:`Subtitle` only checks that
    ``end`` follows ``start``, so a caller can hand us a negative start.
    """
    total_cs = int(round(float(seconds) * 100))
    if total_cs < 0:
        total_cs = 0
    total_s, centis = divmod(total_cs, 100)
    minutes, secs = divmod(total_s, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:d}:{:02d}:{:02d}.{:02d}".format(hours, minutes, secs, centis)


def _parse_opacity(text: str, spec: str) -> float:
    """Parse the ``@ALPHA`` suffix of an ffmpeg colour into 0.0 … 1.0."""
    text = text.strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16) / 255.0
        return float(text)
    except ValueError:
        raise ValueError(
            "cannot read the alpha of colour {!r}: expected a float in "
            "0..1 or a hex byte like 0x80".format(spec)
        )


def _parse_ffmpeg_color(spec) -> tuple:
    """Parse an ffmpeg colour string into ``(r, g, b, opacity)``.

    Accepts ``NAME``, ``#RRGGBB[AA]`` and ``0xRRGGBB[AA]``, each with an
    optional ``@ALPHA`` suffix that overrides a trailing ``AA``. Raises
    :class:`ValueError` for anything else, including the ~120 ffmpeg
    colour names outside :data:`_COLOR_NAMES` — silently rendering the
    wrong colour would be worse than failing before the encode starts.
    """
    text = str(spec).strip()
    base, sep, alpha_text = text.partition("@")
    opacity = _parse_opacity(alpha_text, text) if sep else None
    base = base.strip()
    key = base.lower()
    if key in _COLOR_NAMES:
        red, green, blue = _COLOR_NAMES[key]
        embedded = None
    else:
        if key.startswith("0x"):
            digits = base[2:]
        elif base.startswith("#"):
            digits = base[1:]
        else:
            digits = None
        if digits is None or len(digits) not in (6, 8):
            # Careful with the wording: the *colour* is expressible in
            # ASS perfectly well -- chartreuse is 0x7FFF00 and converts
            # fine. What is missing is the name, because ffmpeg knows
            # ~150 of them and this writer knows a handful. Saying "not
            # supported in ASS" would send the reader looking in the
            # wrong place.
            raise ValueError(
                "unsupported colour name {!r} for ASS output: drawtext "
                "takes any ffmpeg colour name, but the ASS writer only "
                "knows {}. Give it as 0xRRGGBB[AA] or #RRGGBB[AA] "
                "instead. This applies whenever an ASS script is "
                "produced, including a --subtitle-sidecar exported while "
                "drawtext does the rendering.".format(
                    spec, ", ".join(sorted(_COLOR_NAMES))
                )
            )
        try:
            red = int(digits[0:2], 16)
            green = int(digits[2:4], 16)
            blue = int(digits[4:6], 16)
            embedded = (
                int(digits[6:8], 16) / 255.0 if len(digits) == 8 else None
            )
        except ValueError:
            raise ValueError(
                "colour {!r} is not valid hexadecimal".format(spec)
            )
    if opacity is None:
        opacity = 1.0 if embedded is None else embedded
    return red, green, blue, opacity


def _ass_color(spec) -> str:
    """Convert an ffmpeg colour string to an ASS ``&HAABBGGRR`` literal.

    Two traps live in that format: the channels are byte-reversed (BGR,
    not RGB), and the alpha byte is *transparency*, so ``00`` is fully
    opaque and ``FF`` fully transparent — the inverse of ffmpeg's
    ``name@opacity``.
    """
    red, green, blue, opacity = _parse_ffmpeg_color(spec)
    opacity = min(1.0, max(0.0, opacity))
    alpha = 255 - int(round(opacity * 255))
    return "&H{:02X}{:02X}{:02X}{:02X}".format(alpha, blue, green, red)


# Separates a literal backslash from an n/N/h that would otherwise turn
# the pair into an ASS control sequence. Renders nothing at all.
_ZWSP = "\u200b"


def _escape_ass(text: str) -> str:
    r"""Escape subtitle text for an ASS ``Dialogue:`` line.

    Three things need handling, and the rules were checked against what
    libass actually paints rather than against the format's folklore:

    ``{`` and ``}``
        These delimit override blocks (``{\b1}`` turns bold on), so an
        unescaped brace would swallow the text up to the next ``}``.
        Written as ``\{`` / ``\}``, which libass renders as a literal
        brace, dropping the backslash.

    newlines
        Cannot appear inside a Dialogue line at all, and become ``\N``
        (hard break). Note ASS's ``\n`` is a *soft* break that renders
        as a plain space under the default WrapStyle, so it is not a
        substitute.

    ``\``
        Emitted as-is: libass renders an unrecognised ``\x`` as a
        literal backslash followed by ``x``, so ``C:\Users`` survives
        intact. Doubling it — which is what ffmpeg's own escaper in
        ``libavcodec/ass.c`` does — would paint *two* backslashes.

        The exception is a backslash directly before ``n``, ``N`` or
        ``h``, which ASS reads as a soft break, a hard break and a
        non-breaking space. Doubling it does **not** help: rendered
        through libass, ``left\\Nright`` still came back as two lines,
        because the second backslash starts a fresh ``\\N``. A
        zero-width space is inserted between the two characters instead,
        which separates them without drawing anything -- measured
        against a backslash that already renders correctly,
        ``left\\zright`` and ``left\\<zwsp>zright`` are pixel-identical,
        and for each of ``n``, ``N`` and ``h`` the escaped form renders
        on one line at exactly the plain width plus one backslash.

        The cost is a real, if invisible, character in the text and in
        any exported sidecar. ASS has no true escape for a backslash, so
        something has to give; a character nobody can see is a better
        trade than a line break nobody asked for.
    """
    out = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        following = text[index + 1] if index + 1 < length else ""
        if char == "\r" and following == "\n":
            out.append("\\N")
            index += 2
            continue
        if char in "\r\n":
            out.append("\\N")
        elif char == "\\":
            out.append("\\" + _ZWSP if following in ("n", "N", "h") else "\\")
        elif char in "{}":
            out.append("\\" + char)
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _ass_font_name(style) -> str:
    """Family name for the ASS ``Fontname`` field.

    libass looks fonts up by family through fontconfig, so a path is not
    usable here; see :class:`SubtitleStyle`.
    """
    name = style.font_name
    if not name:
        path = style.font_path or _find_font()
        name = os.path.splitext(os.path.basename(path))[0] if path else "Sans"
    if "," in name:
        # Style: is a comma-separated row, so a comma would shift every
        # field after Fontname by one.
        raise ValueError(
            "subtitle font_name must not contain a comma, got {!r}".format(name)
        )
    return name


def ass_fonts_dir(style=None) -> "str | None":
    """Directory to pass libass as ``fontsdir``, or ``None``.

    Lets a font that is not installed system-wide still be found, since
    the ASS script can only name a family, not a file.
    """
    style = style or SubtitleStyle()
    path = style.font_path or _find_font()
    if not path:
        return None
    directory = os.path.dirname(os.path.abspath(path))
    return directory if os.path.isdir(directory) else None


def _ass_style_line(style) -> str:
    primary = _ass_color(style.font_color)
    box = _ass_color(style.box_color)
    fields = [
        _ASS_STYLE_NAME,
        _ass_font_name(style),
        int(style.font_size),
        primary,
        # SecondaryColour only shows in karaoke fills, which we never
        # emit; mirroring the primary keeps it from surprising an editor.
        primary,
        # Under BorderStyle 3 libass fills the box behind the text with
        # OutlineColour, which is what drawtext's boxcolor does.
        box,
        # BackColour is the drop-shadow colour. Shadow is 0 below, so it
        # is never painted.
        box,
        0, 0, 0, 0,          # Bold, Italic, Underline, StrikeOut
        100, 100,            # ScaleX, ScaleY (percent)
        0, 0,                # Spacing, Angle
        3,                   # BorderStyle 3 = opaque box, like box=1
        int(style.border),   # Outline doubles as the box padding at 3
        0,                   # Shadow: SubtitleStyle has no equivalent
        _ASS_ALIGNMENT[style.position],
        _ASS_MARGIN_H,
        _ASS_MARGIN_H,
        _ASS_MARGIN_V[style.position],
        1,                   # Encoding: 1 = default charset
    ]
    return "Style: " + ",".join(str(field) for field in fields)


def build_ass_document(subtitles, style=None, width=1920, height=1080) -> str:
    """Render ``subtitles`` as a complete ASS script.

    ``width`` / ``height`` become ``PlayResX`` / ``PlayResY``. Matching
    them to the video keeps ASS script units equal to pixels, so
    ``SubtitleStyle.font_size`` and the margins mean the same thing they
    do in the drawtext backend; libass rescales the script when they
    differ. :class:`fastgrab.recording.encoder.FfmpegEncoder` always
    passes the real frame size.

    Unlike :func:`build_subtitle_filters` this never returns ``None``:
    libass falls back to a default face when the named font is missing,
    so the ASS backend still renders on a system where no font file was
    discovered.
    

    Two timing differences from the drawtext backend, both measured by
    rendering at 100 fps and counting lit pixels per frame:

    * the end is *exclusive*. For a cue of 0.5 to 1.0, both backends draw
      from frame 0.50, and at exactly 1.00 drawtext still draws while ASS
      has stopped -- one frame's difference, and the format's own
      semantics rather than something to paper over.
    * timestamps are centiseconds, so a cue shorter than 10 ms rounds to
      zero length and can never be drawn. That is refused here rather
      than written out, since the alternative is a subtitle missing from
      both the video and the sidecar with nothing to explain it.
    """
    style = style or SubtitleStyle()
    lines = [
        "[Script Info]",
        "; Generated by fastgrab.recording — plain text, safe to edit.",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "PlayResX: {}".format(int(width)),
        "PlayResY: {}".format(int(height)),
        "",
        "[V4+ Styles]",
        "Format: " + _ASS_STYLE_FORMAT,
        _ass_style_line(style),
        "",
        "[Events]",
        "Format: " + _ASS_EVENT_FORMAT,
    ]
    for sub in subtitles or ():
        start, end = _ass_time(sub.start), _ass_time(sub.end)
        if start == end:
            # ASS timestamps are centiseconds, so a cue shorter than 10 ms
            # rounds to zero length and libass never draws it. Measured:
            # 1.001-1.004 becomes 0:00:01.00,0:00:01.00 and renders
            # nothing at any frame. Emitting it anyway would be a
            # subtitle silently missing from the video and from the
            # sidecar, with nothing to explain why.
            raise ValueError(
                "subtitle {!r} lasts {:.4f}s, which rounds to nothing at "
                "ASS's centisecond resolution ({} to {}), so it would "
                "never be shown. Give it at least 0.01s, or use the "
                "drawtext backend.".format(
                    sub.text, sub.end - sub.start, start, end
                )
            )
        lines.append(
            "Dialogue: 0,{start},{end},{style},,0,0,0,,{text}".format(
                start=start,
                end=end,
                style=_ASS_STYLE_NAME,
                text=_escape_ass(sub.text),
            )
        )
    return "\n".join(lines) + "\n"


def build_ass_filter(path: str, fontsdir: str = None) -> str:
    """Build the ``ass=`` filter that burns ``path`` into the video.

    The filename is escaped twice on purpose — see
    :func:`fastgrab.recording.encoder._escape_filter_value`.
    """
    parts = ["ass=filename=" + _escape_filter_value(_escape_filter_value(path))]
    if fontsdir:
        parts.append(
            "fontsdir=" + _escape_filter_value(_escape_filter_value(fontsdir))
        )
    return ":".join(parts)
