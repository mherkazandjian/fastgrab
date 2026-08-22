"""Capture-loop wrapper that drives :class:`FfmpegEncoder` from a backend."""
import threading
import time

from fastgrab.screenshot import Screenshot

from .clicks import ClickStyle, MouseTracker, draw_cursor, overlay_clicks
from .encoder import FfmpegEncoder, infer_codec


class Recorder:
    """Record a region of the screen to mp4 / webm / gif.

    Draft scope: Linux/X11 only. The default ``backend='x11'`` forces
    the X11 backend even if ``$WAYLAND_DISPLAY`` is set, so the recorder
    behaves predictably under XWayland sessions during development.

    ``title`` and ``overlay_text`` are baked into the encode by ffmpeg
    drawtext filters. ``subtitles`` (a list of
    :class:`fastgrab.recording.subtitles.Subtitle`, styled via
    ``subtitle_style``) are rendered the same way, each during its own
    start/end window.

    ``show_clicks`` overlays a click animation (see
    :class:`fastgrab.recording.clicks.ClickStyle`, passed as
    ``click_style``) on every detected mouse-button press, and
    ``show_cursor`` stamps an emulated arrow pointer at the polled
    pointer position — the XGetImage capture path never includes the
    real cursor sprite (both Linux/X11, require python-xlib).
    """

    def __init__(self, output_path: str, bbox=None, fps: int = 30,
                 codec: str = None, backend: str = "x11",
                 title: str = None, overlay_text: str = None,
                 show_clicks: bool = False, click_style=None,
                 show_cursor: bool = False, cursor_color=(255, 255, 255),
                 cursor_scale: float = 1.0,
                 subtitles=None, subtitle_style=None):
        self.output_path = output_path
        self.bbox = bbox  # (x, y, w, h) or None for fullscreen
        self.fps = fps
        self.codec = codec or infer_codec(output_path)
        self.title = title
        self.overlay_text = overlay_text
        self.show_clicks = show_clicks
        self.click_style = click_style
        self.show_cursor = show_cursor
        self.cursor_color = cursor_color
        self.cursor_scale = cursor_scale
        self.subtitles = subtitles
        self.subtitle_style = subtitle_style
        self._grab = Screenshot(backend=backend)

    def _resolved_bbox(self):
        if self.bbox is not None:
            x, y, w, h = self.bbox
        else:
            sw, sh = self._grab.screensize
            x, y, w, h = 0, 0, sw, sh
        # libx264 (mp4) and libvpx-vp9 (webm) encode to yuv420p, which
        # requires even width/height. Drop the odd pixel rather than
        # padding — sub-pixel difference, encoder-fatal if not handled.
        w -= w % 2
        h -= h % 2
        if w <= 0 or h <= 0:
            raise ValueError(
                "region too small after codec alignment: {!r}".format(
                    self.bbox
                )
            )
        return (x, y, w, h)

    def record(self, duration: float = None,
               stop_event: threading.Event = None,
               on_progress=None, countdown: float = 0,
               on_countdown=None) -> dict:
        """Run the capture loop until ``duration`` elapses or ``stop_event`` fires.

        Returns a small stats dict: ``frames`` (frames captured from the
        screen), ``written_frames`` (frames handed to ffmpeg — captured
        plus duplicates, see below), ``elapsed_seconds``, ``achieved_fps``
        (captured / elapsed, i.e. how fast *capture* actually ran) and
        ``output``. Either bound is sufficient; passing neither records
        until the process is interrupted (Ctrl-C in the CLI), which
        surfaces here as :class:`KeyboardInterrupt`.

        ffmpeg stamps incoming raw frames at the fixed target rate, so
        when capture runs slower than ``fps`` the latest frame is written
        multiple times — once per elapsed tick — keeping the output's
        duration equal to wall-clock time (and subtitle windows correct)
        at the cost of duplicated frames. ``written_frames - frames`` is
        the number of duplicates.

        ``on_progress``, if given, is called after every captured frame as
        ``on_progress(n_frames, elapsed_seconds)`` so a caller can render a
        live counter. It runs inline in the capture loop, so keep it cheap
        and — from a GUI — thread-safe: update a shared value and let the
        UI poll it rather than touching widgets from here.

        ``countdown`` delays the first captured frame by that many seconds
        (ffmpeg isn't even spawned until it elapses), buying time to hide
        or minimise a control window before capture begins. While it runs,
        ``on_countdown(seconds_remaining)`` is called ~10×/s. ``stop_event``
        is honoured during the countdown too: firing it there aborts before
        anything is recorded and returns a zero-frame stats dict.
        """
        bbox = self._resolved_bbox()
        x0, y0, width, height = bbox
        period = 1.0 / float(self.fps)

        encoder = FfmpegEncoder(
            self.output_path, width, height, fps=self.fps, codec=self.codec,
            title=self.title, overlay_text=self.overlay_text,
            subtitles=self.subtitles, subtitle_style=self.subtitle_style,
        )
        tracker = (
            MouseTracker(
                lifetime=(self.click_style or ClickStyle()).lifetime
            )
            if (self.show_clicks or self.show_cursor) else None
        )

        if countdown and countdown > 0:
            cd_deadline = time.monotonic() + countdown
            while True:
                remaining = cd_deadline - time.monotonic()
                if remaining <= 0:
                    break
                if stop_event is not None and stop_event.is_set():
                    # Cancelled before the first frame — nothing was written,
                    # so ffmpeg was never started and there's no file.
                    return {
                        "frames": 0,
                        "written_frames": 0,
                        "elapsed_seconds": 0.0,
                        "achieved_fps": 0.0,
                        "output": self.output_path,
                    }
                if on_countdown is not None:
                    on_countdown(remaining)
                time.sleep(min(0.1, remaining))
            if on_countdown is not None:
                on_countdown(0.0)

        n_captured = 0  # frames grabbed from the screen
        n_written = 0   # frames handed to ffmpeg (captured + duplicates)
        t0 = time.monotonic()
        deadline = (t0 + duration) if duration is not None else None
        try:
            if tracker is not None:
                tracker.start()
            with encoder:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    frame = self._grab.capture(bbox=bbox)
                    if tracker is not None:
                        events = tracker.poll()
                        if self.show_clicks and events:
                            overlay_clicks(
                                frame, events, bbox_origin=(x0, y0),
                                style=self.click_style,
                            )
                        if self.show_cursor and tracker.position is not None:
                            # Drawn last so the pointer sits on top of the
                            # click animation.
                            draw_cursor(
                                frame, *tracker.position,
                                bbox_origin=(x0, y0),
                                color=self.cursor_color,
                                scale=self.cursor_scale,
                            )
                    n_captured += 1
                    # ffmpeg timestamps every raw frame at the fixed target
                    # rate, so if capture + overlays run slower than the
                    # target fps the output would otherwise come out short
                    # and sped up (and subtitle windows would drift). Write
                    # the frame once for every tick that has come due —
                    # duplicating frames when we're behind — so the output
                    # duration tracks wall-clock time. The 1e-6 guards
                    # against float rounding (n * period landing a hair
                    # below the tick) silently dropping a frame captured
                    # exactly on time.
                    now = time.monotonic()
                    due = int((now - t0) / period + 1e-6)
                    while n_written <= due:
                        encoder.write_frame(frame)
                        n_written += 1
                    if on_progress is not None:
                        on_progress(n_captured, now - t0)
                    # Sleep until the next *unwritten* tick rather than
                    # advancing by one period: after a stall n_written has
                    # jumped ahead, and a stale tick would make the loop
                    # spin, capturing frames that `due` then discards.
                    next_tick = t0 + n_written * period
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
        finally:
            if tracker is not None:
                tracker.close()
        elapsed = time.monotonic() - t0
        return {
            "frames": n_captured,
            "written_frames": n_written,
            "elapsed_seconds": elapsed,
            "achieved_fps": (n_captured / elapsed) if elapsed > 0 else 0.0,
            "output": self.output_path,
        }
