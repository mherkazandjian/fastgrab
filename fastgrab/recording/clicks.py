"""Pointer overlays for the recorder: click patterns and an emulated cursor.

The tracker polls the X11 root pointer state via ``python-xlib`` once
per recorded frame, detects 0→1 transitions on the button bitmask, keeps
a small list of recent click events, and remembers the latest pointer
position. :func:`overlay_clicks` draws one of several configurable
patterns (see :class:`ClickStyle`) on the BGRA frame for each event
whose age is below the style's lifetime; :func:`draw_cursor` stamps an
emulated arrow pointer at the polled position — needed because the
XGetImage capture path never includes the real cursor sprite.

python-xlib is an optional runtime dep behind the ``[gui]`` extra; the
import is deferred to :meth:`MouseTracker.start` so the rest of the
recording module stays usable without it.
"""
import time
from dataclasses import dataclass

import numpy


CLICK_LIFETIME = 0.5  # seconds — how long a click ring is shown
RING_R0 = 8           # starting radius in pixels
RING_R1 = 60          # final radius in pixels
RING_THICKNESS = 4    # pixels
# B, G, R — bright cyan reads well on most desktops
RING_COLOR = (255, 200, 0)

CLICK_PATTERNS = ("ring", "concentric", "circle", "crosshair")


@dataclass
class ClickStyle:
    """Visual style for click overlays.

    ``pattern`` is one of :data:`CLICK_PATTERNS`:

    * ``ring`` — a single expanding, fading ring (the original look).
    * ``concentric`` — three phase-offset expanding rings.
    * ``circle`` — a filled disc that fades out in place.
    * ``crosshair`` — a plus-shaped marker that contracts onto the
      click point while fading.

    ``color`` is a ``(B, G, R)`` tuple (frames are BGRA). ``lifetime``
    is how long the animation runs in seconds; ``radius0``/``radius1``
    bound the animation's radii; ``thickness`` is the stroke width for
    ring-based patterns.
    """

    pattern: str = "ring"
    color: tuple = RING_COLOR
    lifetime: float = CLICK_LIFETIME
    radius0: int = RING_R0
    radius1: int = RING_R1
    thickness: int = RING_THICKNESS

    def __post_init__(self):
        if self.pattern not in CLICK_PATTERNS:
            raise ValueError(
                "unknown click pattern {!r}; expected one of {}".format(
                    self.pattern, ", ".join(CLICK_PATTERNS)
                )
            )


# X11 button bits in the pointer mask returned by XQueryPointer.
_BUTTON_BITS = (
    (0x100, "Button1"),  # left
    (0x200, "Button2"),  # middle
    (0x400, "Button3"),  # right
)


class MouseTracker:
    """Edge-detect button presses on the X11 root pointer.

    :meth:`poll` is called once per captured frame; it returns the new
    list of "active" click events (each one a dict with ``x``, ``y``,
    ``t_press``) including any from previous polls that haven't aged
    out yet, and updates :attr:`position` with the latest pointer
    coordinates. The recorder hands the event list to
    :func:`overlay_clicks` and the position to :func:`draw_cursor` —
    when only the cursor is wanted, the returned events are simply
    ignored.

    ``lifetime`` is how long events are kept, in seconds. The recorder
    passes its :class:`ClickStyle` lifetime here so a non-default
    animation duration isn't cut short by the tracker's pruning.
    """

    def __init__(self, lifetime: float = CLICK_LIFETIME):
        self._display = None
        self._root = None
        self._prev_mask = 0
        self._events = []  # list of {"x", "y", "t_press"}
        self._lifetime = lifetime
        self.position = None  # (x, y) screen-absolute, from the last poll

    def start(self):
        try:
            from Xlib import display as xdisplay
        except ImportError as exc:
            raise RuntimeError(
                "show-clicks / show-cursor requires python-xlib — install "
                "with pip install fastgrab[gui]"
            ) from exc
        self._display = xdisplay.Display()
        self._root = self._display.screen().root
        self._prev_mask = 0

    def close(self):
        if self._display is not None:
            try:
                self._display.close()
            except Exception:
                pass
        self._display = None
        self._root = None

    def poll(self) -> list:
        if self._root is None:
            raise RuntimeError("MouseTracker not started")
        p = self._root.query_pointer()
        self.position = (p.root_x, p.root_y)
        now = time.monotonic()
        for bit, _name in _BUTTON_BITS:
            was_down = bool(self._prev_mask & bit)
            is_down = bool(p.mask & bit)
            if is_down and not was_down:
                self._events.append({
                    "x": p.root_x,
                    "y": p.root_y,
                    "t_press": now,
                })
        self._prev_mask = p.mask
        # Drop expired events so the list stays bounded.
        self._events = [
            e for e in self._events
            if (now - e["t_press"]) < self._lifetime
        ]
        return list(self._events)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _blend(frame, mask, y0, y1, x0, x1, color_bgr, alpha):
    """Alpha-blend ``color_bgr`` onto ``frame`` where ``mask`` is set.

    ``mask`` is defined over the sub-region ``frame[y0:y1, x0:x1]`` —
    keeping the blend scoped to a bounding box is what makes these
    overlays cheap at 4K.
    """
    sub = frame[y0:y1, x0:x1]
    inv = 1.0 - alpha
    for c, v in enumerate(color_bgr):
        sub[..., c][mask] = (
            sub[..., c][mask].astype(numpy.float32) * inv
            + v * alpha
        ).astype(numpy.uint8)
    # Alpha channel — leave at full opacity in the BGRA buffer; ffmpeg
    # ignores it when the codec uses yuv420p anyway.
    sub[..., 3][mask] = 255


def _distance_grid(frame, cx, cy, r_outer):
    """Bounding-box slice + squared-distance grid around ``(cx, cy)``."""
    h, w = frame.shape[:2]
    x0 = int(max(0, cx - r_outer - 1))
    y0 = int(max(0, cy - r_outer - 1))
    x1 = int(min(w, cx + r_outer + 2))
    y1 = int(min(h, cy + r_outer + 2))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = numpy.ogrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return x0, y0, x1, y1, d2


def _draw_ring(frame, cx, cy, radius, thickness, color_bgr, alpha):
    """Alpha-blend a filled annulus onto a BGRA frame in place.

    Operates only inside the ring's bounding box for speed — at 4K the
    full-frame mask would dominate the per-frame budget, but the ring
    is at most ~120×120 px so this stays cheap.
    """
    r_outer = radius + thickness / 2.0
    r_inner = max(radius - thickness / 2.0, 0.0)
    box = _distance_grid(frame, cx, cy, r_outer)
    if box is None:
        return
    x0, y0, x1, y1, d2 = box
    mask = (d2 <= r_outer * r_outer) & (d2 >= r_inner * r_inner)
    if mask.any():
        _blend(frame, mask, y0, y1, x0, x1, color_bgr, alpha)


def _draw_disc(frame, cx, cy, radius, color_bgr, alpha):
    """Alpha-blend a filled disc onto a BGRA frame in place."""
    box = _distance_grid(frame, cx, cy, radius)
    if box is None:
        return
    x0, y0, x1, y1, d2 = box
    mask = d2 <= radius * radius
    if mask.any():
        _blend(frame, mask, y0, y1, x0, x1, color_bgr, alpha)


def _draw_crosshair(frame, cx, cy, arm, thickness, color_bgr, alpha):
    """Alpha-blend a plus-shaped marker centred on ``(cx, cy)``."""
    h, w = frame.shape[:2]
    half = max(thickness / 2.0, 0.5)
    # Horizontal and vertical bars, each clipped to the frame.
    for (xa, ya, xb, yb) in (
        (cx - arm, cy - half, cx + arm, cy + half),
        (cx - half, cy - arm, cx + half, cy + arm),
    ):
        x0 = int(max(0, xa))
        y0 = int(max(0, ya))
        x1 = int(min(w, xb + 1))
        y1 = int(min(h, yb + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        mask = numpy.ones((y1 - y0, x1 - x0), dtype=bool)
        _blend(frame, mask, y0, y1, x0, x1, color_bgr, alpha)


def _overlay_pattern(frame, cx, cy, t, style):
    """Draw one animation frame of ``style.pattern`` at progress ``t``."""
    r0, r1 = style.radius0, style.radius1
    alpha = 1.0 - t  # linear fade, shared by all patterns
    if style.pattern == "ring":
        _draw_ring(frame, cx, cy, r0 + (r1 - r0) * t,
                   style.thickness, style.color, alpha)
    elif style.pattern == "concentric":
        # Three rings phase-offset by a third of the lifetime each, so
        # at any moment they sit at different radii.
        for offset in (0.0, 1.0 / 3.0, 2.0 / 3.0):
            phase = (t + offset) % 1.0
            _draw_ring(frame, cx, cy, r0 + (r1 - r0) * phase,
                       style.thickness, style.color, alpha * (1.0 - phase))
    elif style.pattern == "circle":
        _draw_disc(frame, cx, cy, r0 + (r1 - r0) * t, style.color, alpha)
    elif style.pattern == "crosshair":
        # Arm length contracts from radius1 to radius0 as the marker
        # homes in on the click point.
        arm = r1 - (r1 - r0) * t
        _draw_crosshair(frame, cx, cy, arm, style.thickness,
                        style.color, alpha)


def overlay_clicks(frame, events, bbox_origin=(0, 0), now=None,
                   style=None) -> None:
    """Draw the click animation for each event onto ``frame`` in place.

    ``events`` is the list from :meth:`MouseTracker.poll`; coordinates
    are in screen-absolute pixels. ``bbox_origin`` is the top-left of
    the captured region — we subtract it so a click at ``(800, 600)``
    on screen lands at the right pixel inside the captured frame.

    ``style`` is a :class:`ClickStyle`; ``None`` uses the defaults,
    which reproduce the original single expanding ring.
    """
    if not events:
        return
    if style is None:
        style = ClickStyle()
    if now is None:
        now = time.monotonic()
    for e in events:
        age = now - e["t_press"]
        if age < 0 or age >= style.lifetime:
            continue
        t = age / style.lifetime  # 0..1
        cx = e["x"] - bbox_origin[0]
        cy = e["y"] - bbox_origin[1]
        _overlay_pattern(frame, cx, cy, t, style)


# Classic arrow pointer, tip at (0, 0), in a ~12×19 px box. Drawn as an
# outline pass (scaled-up polygon in the outline colour) then a fill
# pass so the pointer reads on both dark and light backgrounds.
_CURSOR_POLY = (
    (0, 0), (0, 16), (4, 12), (7, 18), (9, 17), (6, 11), (11, 11),
)


def _poly_mask(shape, poly):
    """Boolean mask of pixels inside ``poly`` over an array of ``shape``.

    Ray-casting point-in-polygon, vectorised with numpy. ``poly`` is a
    sequence of ``(x, y)`` vertices; the mask has ``shape`` ``(h, w)``.
    """
    h, w = shape
    yy, xx = numpy.mgrid[0:h, 0:w]
    # Sample at pixel centres.
    px = xx + 0.5
    py = yy + 0.5
    inside = numpy.zeros((h, w), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # Edge crosses the horizontal ray to the right of the pixel?
        crosses = ((y1 > py) != (y2 > py))
        if not crosses.any():
            continue
        x_int = x1 + (py - y1) * (x2 - x1) / float(y2 - y1)
        inside ^= crosses & (px < x_int)
    return inside


def draw_cursor(frame, x, y, bbox_origin=(0, 0), color=(255, 255, 255),
                outline=(0, 0, 0), scale=1.0) -> None:
    """Stamp an emulated arrow pointer onto ``frame`` in place.

    ``x``/``y`` are screen-absolute pointer coordinates (e.g.
    :attr:`MouseTracker.position`); ``bbox_origin`` converts them into
    frame coordinates like in :func:`overlay_clicks`. ``color`` and
    ``outline`` are ``(B, G, R)`` tuples.

    This exists because the X11 capture path (``XGetImage``) never
    includes the cursor sprite — without an emulated pointer the mouse
    is invisible in recordings.
    """
    cx = x - bbox_origin[0]
    cy = y - bbox_origin[1]
    h, w = frame.shape[:2]

    def scaled(grow):
        return [
            ((vx - 1) * scale * grow, (vy - 1) * scale * grow)
            for vx, vy in _CURSOR_POLY
        ]

    # Outline pass first (slightly larger polygon), then the fill.
    for poly, rgb in ((scaled(1.18), outline), (scaled(1.0), color)):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0 = int(max(0, cx + min(xs) - 1))
        y0 = int(max(0, cy + min(ys) - 1))
        x1 = int(min(w, cx + max(xs) + 2))
        y1 = int(min(h, cy + max(ys) + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        local = [(vx + cx - x0, vy + cy - y0) for vx, vy in poly]
        mask = _poly_mask((y1 - y0, x1 - x0), local)
        if mask.any():
            _blend(frame, mask, y0, y1, x0, x1, rgb, 1.0)
