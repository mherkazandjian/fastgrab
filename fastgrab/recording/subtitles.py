"""Timed subtitles rendered through ffmpeg's drawtext filter.

A :class:`Subtitle` carries text plus a start/end window in seconds
from the start of the recording; :func:`build_subtitle_filters` turns a
list of them into a comma-joined chain of ``drawtext`` filters (one per
subtitle, gated with ``enable='between(t,start,end)'``) that
:class:`fastgrab.recording.encoder.FfmpegEncoder` appends to its ``-vf``
argument.

Rendering is done entirely by ffmpeg, so there are no new Python
runtime dependencies. Colours are passed through as ffmpeg colour
strings (``white``, ``0xRRGGBB``, ``name@alpha``), which keeps the full
drawtext expressiveness without any parsing on our side.
"""
from dataclasses import dataclass

from .encoder import (
    _escape_drawtext,
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
    """

    font_path: str = None
    font_size: int = 28
    font_color: str = "white"
    box_color: str = "black@0.55"
    border: int = 8
    position: str = "bottom"

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
    # Same parser as the text — a ':' in the path (e.g. Windows drive
    # letters) would break the filter string if left unescaped.
    font = _escape_drawtext(font)
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
