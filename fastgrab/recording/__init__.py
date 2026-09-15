"""Screen recording on top of fastgrab's capture backends.

Draft, Linux/X11 only. Frames come from
:class:`fastgrab.screenshot.Screenshot` (BGRA uint8) and are piped as
rawvideo into a single ``ffmpeg`` subprocess that encodes to mp4 / webm
/ gif depending on the output file extension.

Public surface:

* :class:`Recorder`      — high-level capture-loop wrapper
* :class:`FfmpegEncoder` — thin subprocess wrapper, useful on its own
* :func:`infer_codec`    — extension → codec mapping
* :class:`ClickStyle`    — click-overlay pattern / colour / timing
* :class:`Subtitle`      — one timed subtitle line
* :class:`SubtitleStyle` — font / colour / placement for subtitles
* :class:`BlurStyle`     — how blurred/redacted regions are obscured
  (re-exported from :mod:`fastgrab.effects`, which is core, not
  recording-specific)
* :data:`SUBTITLE_BACKENDS` — subtitle renderers (``drawtext``, ``ass``)
* :func:`build_ass_document` — render subtitles as an ASS script

The module imports cleanly without ``ffmpeg`` on PATH; the check is
deferred to :meth:`FfmpegEncoder.start` so callers get a clear error
only when they actually try to encode.
"""
from fastgrab.effects import BlurStyle

from .clicks import ClickStyle
from .encoder import SUBTITLE_BACKENDS, FfmpegEncoder, infer_codec
from .recorder import Recorder
from .subtitles import Subtitle, SubtitleStyle, build_ass_document

__all__ = [
    "Recorder", "FfmpegEncoder", "infer_codec",
    "ClickStyle", "Subtitle", "SubtitleStyle", "BlurStyle",
    "SUBTITLE_BACKENDS", "build_ass_document",
]
