---
name: fastgrab-user
description: Expert guidance for *using* the fastgrab Python package — high-frame-rate screen capture into a numpy array (Linux X11/Wayland, Windows, macOS) plus the opt-in ffmpeg-based screen recorder. Load this when writing, debugging, or explaining code that installs or calls fastgrab. Not for contributing to fastgrab itself (see fastgrab-dev).
---

# fastgrab — user skill

You are an expert in **using** `fastgrab`, a small Python package whose
whole point is a two-line, dependency-light screen capture that returns a
numpy array:

```python
from fastgrab import screenshot
img = screenshot.Screenshot().capture()   # numpy.ndarray, uint8, shape (H, W, 4)
```

Everything below is what you need to install it correctly, call it
correctly, avoid its known footguns, and diagnose the common failures.
When in doubt, favour the minimal core API and keep the user's install
surface small — that is the project's design philosophy.

## 1. Facts at a glance

| item | value |
|---|---|
| package / import | `pip install fastgrab` → `from fastgrab import screenshot` |
| Python | `>= 3.10` (no Python 2) |
| runtime deps (default wheel) | `numpy >= 1.26` only |
| return type | `numpy.ndarray`, dtype `uint8`, shape `(height, width, 4)` |
| **channel order** | **BGRA** — `img[..., 0]`=B, `1`=G, `2`=R, `3`=A (alpha usually 0) |
| typical speed | ~200 fps at 1080p, >800 fps at 360p, ~20 fps at 4K (modern CPU) |
| platforms | Linux (X11 default; Wayland via extra), Windows 10/11, macOS |
| license | GPLv3 |
| repo | https://github.com/mherkazandjian/fastgrab |

## 2. Installing — per platform

```bash
pip install fastgrab                     # all platforms, core only
pip install fastgrab[wayland]            # + wlroots Wayland backend (pywayland)
pip install fastgrab[gui]                # + python-xlib & Pillow (click/cursor overlays, --gui)
pip install fastgrab[wayland-portal]     # xdg-desktop-portal backend — STUB, not functional yet
pip install git+https://github.com/mherkazandjian/fastgrab.git   # latest from source
```

Platform prerequisites:

- **Linux / X11** — a C extension is compiled on install. Needs a C
  compiler and X11 headers: `gcc`, `libx11-dev` (Debian/Ubuntu) /
  `libX11-devel` (Fedora). Runtime: `libX11`, `libgomp1`. If
  `pip install` fails on Linux, a missing `libx11-dev` or `gcc` is the
  cause ~90% of the time.
- **Linux / Wayland** — `pip install fastgrab[wayland]`. Works on
  **wlroots** compositors only (Sway, Hyprland, river, niri, cage) via
  `wlr-screencopy-v1`. GNOME and KDE do not expose that protocol; the
  portal backend for them is currently a `NotImplementedError` stub. On
  GNOME/KDE Wayland, fastgrab falls back to X11 through XWayland, which
  only sees X11 clients — native Wayland windows appear black/missing.
- **Windows 10/11** — nothing extra. Pure `ctypes` over GDI `BitBlt`.
- **macOS** — nothing extra. Pure `ctypes` over CoreGraphics
  `CGDisplayCreateImage`. macOS 10.15+ requires **Screen Recording
  permission** for the process (System Settings → Privacy & Security →
  Screen Recording). Without it `capture()` raises
  `RuntimeError("CGDisplayCreateImage returned NULL — likely a Screen
  Recording (TCC) permission denial ...")`; on some macOS versions you may
  instead get an image showing only the wallpaper. Either way: grant the
  permission to the terminal/app and restart it.

## 3. Core API — `fastgrab.screenshot.Screenshot`

```python
from fastgrab import screenshot

grab = screenshot.Screenshot(backend=None)   # backend: None | 'x11' | 'wlr' | 'portal' | 'windows' | 'macos'

grab.screensize          # (width, height) of the primary screen — cached after first read
img = grab.capture()     # full screen
img = grab.capture(bbox=(x, y, w, h))   # sub-region: top-left corner + size, in pixels
```

Semantics you must get right:

- `bbox` is `(x0, y0, width, height)` — **not** `(left, top, right, bottom)`.
  `x0 + width` and `y0 + height` must not exceed the screen size or
  `capture()` raises `ValueError("bbox is outside the screen boarders ...")`.
- **The returned array is a reused internal buffer.** `capture()` returns
  the *same* ndarray object on every call while the requested size is
  unchanged. If you keep frames (a list, a queue, a thread), you must
  `img.copy()` — otherwise every stored frame silently becomes the latest
  one. This is the single most common fastgrab bug.
- A new buffer is allocated only when the bbox size changes. For max
  throughput keep one `Screenshot` instance alive and capture the same
  size repeatedly; don't construct `Screenshot()` per frame.
- Capture covers the **primary screen only**. Multi-monitor spanning is
  not implemented on any backend.
- The **mouse cursor is never part of the captured pixels** on X11
  (`XGetImage` does not include the cursor sprite). The recorder has an
  emulated cursor for this reason (section 5).
- Backend auto-detection (`backend=None`): Windows → `windows`; macOS →
  `macos`; Linux → if `$WAYLAND_DISPLAY` is set try `wlr` then `portal`,
  else if `$DISPLAY` is set use `x11`; otherwise
  `RuntimeError("no usable display server detected (need DISPLAY or WAYLAND_DISPLAY)")`.
- Forcing a backend whose extra is missing raises
  `RuntimeError("backend 'wlr' requires the wayland extra: pip install fastgrab[wayland]")`
  — surface that hint verbatim to users. Unknown names raise `ValueError`.

### Converting the BGRA array

```python
rgb  = img[..., 2::-1]               # (H, W, 3) RGB view, no copy — for matplotlib, PIL, imageio
bgr  = img[..., :3]                  # (H, W, 3) BGR view — OpenCV's native order, no conversion needed
gray = img[..., :3].mean(axis=2).astype("uint8")   # quick luminance

from PIL import Image
Image.fromarray(img[..., 2::-1]).save("shot.png")          # PIL wants RGB

import cv2
cv2.imwrite("shot.png", img[..., :3])                        # cv2 wants BGR
cv2.imshow("live", img)                                      # BGRA displays fine too

import matplotlib.pyplot as plt
plt.imshow(img[..., 2::-1]); plt.show()
```

Don't use `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` — the data is already
BGR(A). If colours look swapped (skin tones blue), the user is treating
the array as RGB.

### Idiomatic patterns

```python
# Capture loop at a fixed rate, copying frames you keep
import time
from fastgrab import screenshot

grab = screenshot.Screenshot()
frames = []
period = 1 / 30
t_next = time.perf_counter()
while len(frames) < 300:
    frames.append(grab.capture().copy())      # .copy() is mandatory here
    t_next += period
    time.sleep(max(0.0, t_next - time.perf_counter()))
```

```python
# Benchmark raw capture fps for a region
import time
grab = screenshot.Screenshot()
w, h = grab.screensize
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 2.0:
    grab.capture((0, 0, w // 2, h // 2)); n += 1
print(n / (time.perf_counter() - t0), "fps")
```

```python
# Pick a backend explicitly and fail loudly
try:
    grab = screenshot.Screenshot(backend="wlr")
except RuntimeError as exc:        # missing extra, or no compositor support
    print(exc); raise SystemExit(1)
```

## 4. Performance guidance

- fastgrab is fast because it does **one memcpy into a preallocated numpy
  buffer** — no PIL, no PNG encode, no per-call allocation. Anything the
  user adds on top (`.copy()`, colour conversion, encoding) will dominate.
- Smaller `bbox` → proportionally faster. Capture only what you need.
- Reuse one `Screenshot` object. Constructing one opens a display
  connection / device context.
- Throughput is memory-bandwidth bound; 4K at ~20 fps is expected, not a
  bug.
- For comparison at 1080p on the reference machine: fastgrab 200 fps,
  python-mss 180, autopy 34, pyautogui 8, pyscreenshot 4.

## 5. Screen recording (opt-in, Linux/X11 only, draft)

`fastgrab.recording` pipes captured frames as rawvideo into a single
`ffmpeg` subprocess. Requirements: **`ffmpeg` on `PATH`**; X11 (`DISPLAY`
set); `pip install fastgrab[gui]` for click/cursor overlays and the
`--gui` selector (which also needs `tkinter`). The module imports cleanly
without ffmpeg — the error only appears when encoding starts.

### CLI — `fastgrab-record`

Exactly one capture target is required: `--fullscreen`, `--region X,Y,W,H`,
or `--gui`. Output format is inferred from the extension: `.mp4`
(libx264), `.webm` (libvpx-vp9), `.gif`.

```bash
fastgrab-record --fullscreen --duration 10 -o demo.mp4
fastgrab-record --region 100,100,1280,720 --fps 60 -o clip.webm      # Ctrl-C to stop when no --duration
fastgrab-record --fullscreen --countdown 3 --title "My demo" --overlay-text "v1.2" -o demo.mp4
fastgrab-record --fullscreen -o demo.mp4 \
    --show-clicks --click-style concentric --click-color 255,200,0 --click-lifetime 0.6 \
    --show-cursor \
    --subtitle "0.5-3.0:Hello world" --subtitle "4.0-6.5:Second line" \
    --subtitle-font /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
    --subtitle-fontsize 32 --subtitle-color yellow --subtitle-box-color black@0.6 \
    --subtitle-position bottom
fastgrab-record --gui                      # drag-select a region, then a config dialog
fastgrab-record --print-xbindkeys          # snippet to bind --gui to a hotkey (e.g. Print)
```

Flag reference:

| flag | meaning |
|---|---|
| `-o/--output PATH` | required unless `--gui`; `.mp4` / `.webm` / `.gif` |
| `--fullscreen` / `--region X,Y,W,H` / `--gui` | capture target (pick one); W,H ≥ 2 |
| `--fps N` | target fps, default 30 |
| `--duration S` | stop after S seconds; omit → run until Ctrl-C / SIGTERM (file is finalised cleanly) |
| `--countdown S` | delay before the first frame (ffmpeg not even spawned until it ends) |
| `--title TEXT` | drawtext, top-centre, first 3 s |
| `--overlay-text TEXT` | drawtext watermark, top-right, whole clip |
| `--show-clicks` | animate mouse clicks (`[gui]` extra) |
| `--click-style` | `ring` (default) / `concentric` / `circle` / `crosshair` |
| `--click-color B,G,R` | note **BGR** order, e.g. `255,200,0`; default cyan |
| `--click-lifetime S` | animation length, default 0.5 |
| `--show-cursor` | stamp an emulated arrow at the pointer (`[gui]` extra) |
| `--subtitle START-END:TEXT` | repeatable; seconds, e.g. `1.5-4.0:Hello` |
| `--subtitle-font PATH` | default `$FASTGRAB_FONT`, else bundled DejaVu search paths |
| `--subtitle-fontsize N` / `--subtitle-color` / `--subtitle-box-color` / `--subtitle-position top\|bottom` | ffmpeg colour strings: `white`, `0xRRGGBB`, `red@0.8` |
| `--backend x11` | only `x11` is accepted in this draft |

### Python API

```python
from fastgrab.recording import Recorder, ClickStyle, Subtitle, SubtitleStyle, FfmpegEncoder, infer_codec

rec = Recorder(
    output_path="demo.mp4",          # codec inferred; or codec="mp4"|"webm"|"gif"
    bbox=(0, 0, 1280, 720),          # None → fullscreen
    fps=30,
    backend="x11",
    title="Demo", overlay_text="v1",
    show_clicks=True, click_style=ClickStyle(pattern="ring", color=(255, 200, 0), lifetime=0.5, thickness=4),
    show_cursor=True, cursor_color=(255, 255, 255), cursor_scale=1.0,
    subtitles=[Subtitle(text="Hello", start=0.5, end=3.0)],
    subtitle_style=SubtitleStyle(font_path=None, font_size=28, font_color="white",
                                 box_color="black@0.55", border=8, position="bottom"),
)
stats = rec.record(duration=5.0)                 # or stop_event=threading.Event(), countdown=3,
                                                 #    on_progress=lambda n, t: ..., on_countdown=lambda s: ...
stats  # {'frames': captured, 'written_frames': captured + duplicates,
       #  'elapsed_seconds': T, 'achieved_fps': captured / T, 'output': path}
```

- `record()` with neither `duration` nor `stop_event` runs until
  `KeyboardInterrupt`. Prefer a `threading.Event` from your own code.
- `on_progress(n_frames, elapsed)` runs inline in the capture loop — keep
  it cheap and don't touch GUI widgets from it.
- Width/height are rounded **down to even** (yuv420p requirement); a
  region that rounds to 0 raises `ValueError`.
- When capture is slower than `fps`, the last frame is written again for
  every missed tick so the clip's duration still matches wall-clock and
  subtitle timings stay correct. `stats["frames"]` / `achieved_fps` report
  real capture speed; `stats["written_frames"] - stats["frames"]` is the
  duplicate count, and the CLI summary prints `N duplicated to hold F fps`
  when it is non-zero.
- `FfmpegEncoder(output_path, width, height, fps=30, codec=None, ...)` is
  usable standalone as a context manager: `start()`, `write_frame(bgra)`,
  `close()`. `write_frame` rejects frames whose shape isn't `(H, W, 4)`.

## 6. Troubleshooting — symptom → cause → fix

| symptom | cause | fix |
|---|---|---|
| `pip install` fails compiling on Linux, mentions `X11/Xlib.h` or `gcc` | missing build deps | `apt install gcc libx11-dev` (or distro equivalent), retry |
| `ImportError: ... _linux_x11` | C extension not built / wrong Python ABI | reinstall in the same interpreter: `pip install --force-reinstall --no-binary :all: fastgrab` |
| `RuntimeError: no usable display server detected` | neither `DISPLAY` nor `WAYLAND_DISPLAY` set (ssh, cron, container) | run under a display, or `xvfb-run python script.py` for headless work |
| `RuntimeError: backend 'wlr' requires the wayland extra` | forced `wlr` without pywayland | `pip install fastgrab[wayland]` |
| Wayland on GNOME/KDE: black frames or only some windows | wlr protocol unavailable; XWayland fallback sees X11 clients only | no fix yet — portal backend is stubbed; run the X11 session or a wlroots compositor |
| `NotImplementedError` from portal backend | `[wayland-portal]` is a placeholder | same as above |
| macOS: `RuntimeError: CGDisplayCreateImage returned NULL — likely a Screen Recording (TCC) permission denial`, or wallpaper-only image | Screen Recording permission not granted | System Settings → Privacy & Security → Screen Recording → enable the terminal/app, restart it |
| colours look wrong (blue skin, orange sky) | array treated as RGB | it's BGRA — use `img[..., 2::-1]` for RGB |
| every stored frame is identical | buffer reuse | `.copy()` each frame you keep |
| `ValueError: bbox is outside the screen boarders` | `x+w` or `y+h` > screen size, or bbox given as (l,t,r,b) | use `(x, y, w, h)` within `Screenshot().screensize` |
| `fastgrab-record` → error about ffmpeg | `ffmpeg` not on PATH | install ffmpeg (`apt install ffmpeg`, `brew install ffmpeg`) |
| `--show-clicks` / `--show-cursor` error about python-xlib | `[gui]` extra missing | `pip install fastgrab[gui]` |
| subtitles / title silently missing from video | no usable font found | set `FASTGRAB_FONT=/path/to/font.ttf` or pass `--subtitle-font` |
| `--gui` fails importing tkinter | system Python without Tk | install `python3-tk` (Debian/Ubuntu) or equivalent |
| recorder `ValueError: region too small after codec alignment` | W or H < 2 after even-rounding | use a larger region |
| cursor not in screenshots | by design on X11 | use the recorder's `--show-cursor`, or composite your own |

## 7. Known limitations (state these plainly, don't work around silently)

- Primary screen only; no multi-monitor, no per-window capture.
- Cursor excluded from captured pixels (X11); recorder emulates it.
- Recording is Linux/X11 only and marked *draft*.
- Wayland: wlroots compositors only; GNOME/KDE portal backend is a stub.
- Alpha channel is not meaningful (typically 0) — don't rely on it.
- Wheels for Linux are built from source on install (C extension); there is
  no manylinux binary wheel, hence the compiler requirement.

## 8. How to answer well

- Lead with the two-line API; only add extras / recording when asked.
- Always mention `.copy()` when the user stores or threads frames.
- Always mention BGRA when the user displays, saves, or feeds frames to
  another library.
- Give platform-specific prerequisites before suggesting
  `pip install fastgrab` on Linux.
- Don't invent features: no multi-monitor, no window capture, no
  Windows/macOS recording, no portal support.
