---
name: fastgrab-dev
description: Expert guidance for *contributing to* the fastgrab repository — its mission (two-line numpy capture API, dependency-light default wheel), backend architecture, Poetry + C-extension build, docker-compose-only test workflow, CI matrix, and the non-obvious gotchas. Load this when modifying fastgrab's source, tests, build, or CI. For merely using the package, see fastgrab-user.
---

# fastgrab — developer skill

You are an expert contributor to `fastgrab`
(https://github.com/mherkazandjian/fastgrab). Use this when reading,
modifying, reviewing, or extending the repository itself.

## 1. The mission — read this before changing anything

fastgrab's core value is that a plain `pip install fastgrab` yields a
working two-line API on every supported platform:

```python
from fastgrab import screenshot
img = screenshot.Screenshot().capture()   # numpy uint8, (H, W, 4), BGRA
```

**Everything else is optional and secondary.** The default wheel must stay
a minimal core with **`numpy` as its only runtime dependency**. Additional
functionality — alternative backends, recording, GUI tooling, format
converters — is **opt-in via pip extras** (`pip install fastgrab[<extra>]`)
and must never be required on the base import path.

Why this is load-bearing: a large share of the project's open issues are
install/build failures across distros and Python versions. A tiny,
dependency-light default surface is the simplest defence.

### Decision checklist for any new feature

1. Does it belong behind an extra or in the core? **Default to extra.**
2. Does it add a runtime dependency to the default wheel? If yes it almost
   certainly belongs behind an extra.
3. Is it another OS backend? Pure-Python ones (ctypes against an OS API)
   ship in tree without an extra because they have zero runtime deps —
   that's how `windows.py` and `macos.py` landed. Backends needing a
   non-stdlib library (`pywayland`, `dbus-next`, `pyobjc`) go behind an
   extra, imported lazily, with an install hint in the error message.
4. Does it touch `fastgrab/__init__.py` or `fastgrab/screenshot.py`?
   Verify the two-line API still works unchanged on every platform.
5. New runtime deps and new code paths in the default install are a tax
   against the mission. Justify them explicitly or ship them as an extra.

## 2. Repository map

```
pyproject.toml              Poetry project; single source of truth (deps, extras, scripts, markers)
build.py                    Poetry build hook: compiles the libX11 C extension on Linux only
poetry.lock
Makefile                    host targets; `make <target> docker=1` routes through docker compose
docker-compose.yml          services: dev (VNC desktop), test, test-wayland, benchmark
docker/Dockerfile           python:3.11-slim + gcc/libx11-dev/xvfb/cage/ffmpeg/poetry/pytest/...
docker/*.sh                 entrypoint (in-place ext build), start-desktop, start-wayland-desktop
.github/workflows/test.yml  CI: x11 (required), wayland / windows / macos (fail-soft)
fastgrab/
  __init__.py               version/author metadata only — keep it import-cheap
  metadata.py
  screenshot.py             Screenshot: buffer owner, bbox validation, dispatch to backend
  backends/
    __init__.py             _resolve_backend(name) / _autodetect()
    base.py                 BaseBackend ABC: resolution(), bytes_per_pixel(), screenshot(x, y, img)
    x11.py                  shim over fastgrab._linux_x11 (C ext). Default on Linux/X11.
    wlr.py                  wlr-screencopy-v1 via pywayland.  [wayland] extra
    portal.py               xdg-desktop-portal + PipeWire.    [wayland-portal] extra — STUB
    windows.py              Win32 BitBlt + CreateDIBSection via ctypes. No extra.
    macos.py                CoreGraphics CGDisplayCreateImage via ctypes. No extra.
    protocols/              generated pywayland bindings + the .xml protocol (shipped in wheel)
  linux_x11/screenshot.c    the C extension source (XGetImage → memcpy into the numpy buffer)
  recording/                opt-in recorder (ffmpeg on PATH, X11 only)
    __init__.py             public: Recorder, FfmpegEncoder, infer_codec, ClickStyle, Subtitle, SubtitleStyle
    recorder.py             capture loop; even-dimension rounding; countdown; stop_event
    encoder.py              ffmpeg argv builder, drawtext filters, font discovery ($FASTGRAB_FONT)
    clicks.py               ClickStyle, MouseTracker (python-xlib), overlay_clicks, draw_cursor — pure numpy
    subtitles.py            Subtitle, SubtitleStyle, build_subtitle_filters → drawtext chain
    cli.py                  `fastgrab-record` argparse entry point
    gui/                    tkinter region selector, config dialog, progress window (lazy imports)
tests/                      see section 5
examples/                   single_screenshot.py, low_level_api_screenshot.py, benchmark.py, record_demo.py
```

## 3. Architecture contract

- `Screenshot(backend=None)` resolves a `BaseBackend` once. It lazily
  allocates a `(H, W, 4)` uint8 buffer and **reuses it** across calls
  while the bbox size is unchanged; it reallocates only on size change.
  `capture()` returns that internal buffer (callers must copy to keep
  frames). Do not change this — it is the source of the speed.
- `bbox` is `(x0, y0, width, height)`; `check_bbox` raises `ValueError`
  when `x0+w` or `y0+h` exceeds `screensize`. `screensize` is cached.
- **All backends fill the buffer in BGRA byte order** (B=0, G=1, R=2, A=3
  on little-endian). This is the public numpy contract; integration tests
  assert it. A backend that natively produces RGBA must swizzle.
- `BaseBackend.screenshot(x, y, img)` receives the pre-allocated array;
  the requested size is implied by `img.shape`. Write the whole buffer.
- Auto-detection (`backends/__init__.py:_autodetect`): `sys.platform`
  first (`win32` → windows, `darwin` → macos); on Linux, if
  `$WAYLAND_DISPLAY` try `wlr` then `portal` (each wrapped in a broad
  `except Exception` so a missing extra or an unsupported compositor falls
  through), else `$DISPLAY` → x11, else `RuntimeError`.
- Explicit `backend='wlr'|'portal'` raises `RuntimeError` with the exact
  `pip install fastgrab[...]` hint on `ImportError`. Keep that pattern for
  any new optional backend.
- The C extension `fastgrab._linux_x11` exposes `resolution()`,
  `bytes_per_pixel()`, `screenshot(x, y, img)`. It does a single `memcpy`
  from `XImage->data`; it links `X11` and `gomp`. Every entry point checks
  `XOpenDisplay`/`XGetImage` for NULL and raises `RuntimeError` (before
  0.3.0 an unreachable `DISPLAY` segfaulted); images are released with
  `XDestroyImage`.

## 4. Build system

- **Poetry-managed**, `poetry-core` is the PEP 517 backend. There is **no
  `setup.py`**; `pip install .` and `pip install -e .` work without Poetry
  installed. Poetry itself is only for dependency management
  (`poetry add`, `poetry lock`).
- `build.py` is wired in via `build = "build.py"` in `[tool.poetry]`.
  `build(setup_kwargs)` injects the `Extension` **only when
  `sys.platform == "linux"`**; Windows/macOS get pure-Python wheels.
  `python build.py` runs `build_ext --inplace` so the `.so` lands in-tree —
  the docker entrypoint uses this because poetry-core's editable install
  does not place compiled extensions in-tree.
- `[build-system] requires` pins `numpy>=2.0` for headers (binaries built
  against 2.x run on 1.x per NumPy's guidance; runtime floor stays
  `>=1.26`). Don't "fix" this apparent mismatch.
- Extras (`[tool.poetry.extras]`): `wayland` (pywayland),
  `wayland-portal` (dbus-next, pipewire-python), `wayland-all`, `gui`
  (python-xlib, Pillow). Console script: `fastgrab-record`.
- Build deps on Linux: `gcc`/`build-essential`, `libx11-dev`, numpy
  headers. Runtime: `libX11`, `libgomp1`. The protocol `.xml`/`.py` under
  `backends/protocols/` must stay in the `include` list for the wheel.
- Gotcha: the extension compiles with `-mtune=native`. Wheels built on one
  machine are not guaranteed optimal/portable elsewhere — this is one
  reason there are no prebuilt Linux binary wheels. Don't publish a
  locally built Linux wheel as if it were manylinux.
- Release/publish is **manual**; CI never publishes. Version lives in
  `pyproject.toml` (`0.3.0.dev0` at time of writing) and is surfaced via
  `fastgrab.metadata` / `fastgrab.__version__`.

## 5. Development workflow — docker compose only

**Never run the build, tests, or X11/Wayland commands on the host.** The
dev container is the canonical reproducible environment. If a contributor
hits an install issue, the first question is "does it work in
`docker compose run --rm test`?".

```bash
docker compose run --rm test          # X11 unit + integration tests under xvfb-run (1280x1024)
docker compose run --rm test-wayland  # wlr-screencopy tests under headless cage
docker compose up -d dev              # virtual desktop (fluxbox) with VNC on host :5901
docker compose exec dev bash          # shell in the running desktop container
docker compose exec dev python examples/record_demo.py
docker compose run --rm benchmark     # capture-fps sweep on a 3840x2160 virtual screen
make help                             # Makefile targets; append docker=1 to route through compose
make test docker=1 / make build docker=1 / make lock docker=1 / make clean docker=1
```

Run a subset of tests inside the container:

```bash
docker compose run --rm --entrypoint bash test -c \
  'xvfb-run -a -s "-screen 0 1280x1024x24" python tests/_run_pytest_clean_exit.py -v -k bbox tests'
```

The repo is bind-mounted at `/app`, so edits on the host are visible
immediately; the entrypoint rebuilds the C extension in place on start.

## 6. Tests

| file | scope |
|---|---|
| `tests/test_screenshot.py` | cross-platform API contract: shape, dtype, bbox validation, buffer reuse / reallocation. Runs everywhere (incl. Windows/macOS CI). |
| `tests/test_x11_lowlevel.py` | direct `fastgrab._linux_x11` exercises; module-level skip on non-Linux. |
| `tests/test_integration.py` | `@pytest.mark.integration` — pixel-level: paints the X root via python-xlib `XFillRectangle`, asserts captured BGRA bytes. |
| `tests/test_integration_wlr.py` | `@pytest.mark.wayland` — pixel-level via the `wayland_painter.py` kiosk client under headless `cage`. |
| `tests/test_recording.py` | codec inference, ffmpeg argv/drawtext, click patterns, cursor stamping, subtitles, CLI parsing; a few `skipif(not X11)` recorder smoke tests that really invoke ffmpeg. |
| `tests/conftest.py` | `paint_root` (X11) and `paint_wayland` fixtures; autouse `require_some_display` (no-op on Windows/macOS); `_x11_available()` / `_wayland_available()` helpers; `FASTGRAB_TEST_BACKEND` env selects the backend under test. |
| `tests/_run_pytest_clean_exit.py` | runs pytest then `os._exit` — bypasses pywayland cffi finalizers segfaulting at interpreter shutdown. Use it instead of bare `pytest` in containers. |

Markers are declared in `pyproject.toml` (`integration`, `wayland`). The
compose `test` service runs `-m "not wayland"`; `test-wayland` runs
`-m wayland`; Windows/macOS CI runs `-m "not wayland and not integration"`.

Test-writing rules:

- Pixel tests must draw with a foreground GC + `XFillRectangle` (see
  `paint_root`). **`xsetroot -solid`, `feh --bg-*`, or any
  background-pixmap approach is invisible to `XGetImage` on bare Xvfb** —
  it silently yields all-zero captures.
- Assert channel order explicitly (`img[y, x, 0]` is blue, `2` is red).
- Recording tests that spawn ffmpeg must be `skipif` on missing X11 and
  write into `tmp_path`.
- Overlay/subtitle logic is pure numpy / pure string building — test it
  without a display.
- Keep `test_screenshot.py` platform-agnostic; it is the only suite that
  runs on the Windows and macOS runners.

## 7. CI (`.github/workflows/test.yml`)

Four jobs on every push and PR:

- `x11` — ubuntu-latest, **required**: `docker compose run --rm test`
- `wayland` — ubuntu-latest, `continue-on-error` (headless wlroots flaky)
- `windows` — windows-latest, `continue-on-error`: `pip install .` then
  `pytest -m "not wayland and not integration"`
- `macos` — macos-latest, `continue-on-error`: same as windows

Fail-soft jobs graduate to required once their pass rate is steady. There
are no release/publish steps in CI.

## 8. Conventions and gotchas

- **BGRA everywhere.** To hand RGB to something, slice `img[..., 2::-1]`.
  Click colours in `ClickStyle.color` and `--click-color` are also BGR.
- **Lazy-import optional deps** (`pywayland`, `python-xlib`, `tkinter`,
  `Pillow`) inside the function/class that needs them, and wrap with an
  error carrying the `pip install fastgrab[extra]` hint. The top-level
  `fastgrab.recording` must import without ffmpeg, Tk or python-xlib
  present; ffmpeg is checked only in `FfmpegEncoder.start()`.
- Recording overlays are **pure numpy or ffmpeg filter strings** — no
  Pillow/OpenCV at runtime in the core recording path. Keep it that way.
- Codec alignment: libx264 / libvpx-vp9 use yuv420p → width and height
  are rounded **down to even** in `Recorder._resolved_bbox`; the CLI
  rejects W/H < 2 up front.
- ffmpeg is fed rawvideo at a fixed `-framerate`; slow capture means the
  last frame is written repeatedly so wall-clock duration and subtitle
  windows stay correct. Don't "optimise" this away without replacing the
  timing model.
- Ctrl-C handling in the CLI sets a `threading.Event` rather than raising
  mid-write, so ffmpeg finalises the container. Preserve this.
- `Screenshot.check_bbox` error text says "boarders" (sic). Tests may
  match on it — don't silently rewrite user-visible strings without
  grepping tests.
- `fastgrab/linux_x11/cmake-build-debug/` and `.idea/` are legacy IDE
  artefacts; ignore them, don't build on them.
- The X11 capture never contains the cursor sprite; that is why
  `draw_cursor` exists. Don't file it as a bug.
- Multi-monitor, per-window capture, a functional portal backend, a macOS
  ScreenCaptureKit backend and manylinux wheels are **known open
  follow-ups**, not regressions.

## 9. Recipes

### Add a new capture backend

1. Create `fastgrab/backends/<name>.py` subclassing `BaseBackend`;
   implement `resolution()`, `bytes_per_pixel()` (4), and
   `screenshot(x, y, img)` filling `img` in BGRA.
2. If it needs a non-stdlib library: add it as an optional dependency and
   a new extra in `pyproject.toml`, import it lazily, and add a named
   branch in `_resolve_backend` raising `RuntimeError` with the install
   hint on `ImportError`. Pure-ctypes backends need no extra.
3. Wire it into `_autodetect` in the right order for its platform.
4. Add a pixel-level integration test that paints a known colour and
   asserts BGRA bytes; add a compose service + CI job if it needs a new
   display stack (mirror `test-wayland`).
5. Update the README platform list and the backend docstring in
   `backends/__init__.py`. Re-run `make lock docker=1`.
6. Prove the two-line API is unchanged: `docker compose run --rm test`.

### Add a recording overlay / CLI flag

1. Implement as pure numpy (frame-side, in `clicks.py`-style modules) or
   as an ffmpeg filter string (`encoder.py` / `subtitles.py`).
2. Expose it on `Recorder.__init__`, then on `cli.py:build_parser` with a
   validating `type=` callable (see `_parse_region`, `_parse_bgr`,
   `_parse_subtitle`) and wire it in `main()`.
3. Unit-test the pure logic headless; add one `skipif(not X11)` smoke test
   only if the encode path changes.
4. Document the flag in README under "Screen recording".

### Validate the install story (release prep / install-issue triage)

Use a **fresh minimal base image** (`python:3.11-slim`, `python:3.12-slim`,
`ubuntu:24.04`, `fedora:latest`), install only what the README claims with
`--no-install-recommends` (`build-essential python3-dev libx11-dev libgomp1
xvfb xauth`), and install from a **clean `git archive` export** — never
from a copy or mount of the working tree, which carries gitignored
`build/` objects and prebuilt `_linux_x11*.so` that setuptools silently
reuses (a whole "green" matrix once never compiled a line). Use
`pip install -v` and require a `gcc … screenshot.c` line in the log. Wait
for `/tmp/.X11-unix/X99` before capturing (Fedora's Xvfb takes seconds),
then run the two-line API and expect `(480, 640, 4)` for a `640x480x24`
screen. **Do not use `docker/Dockerfile` for this** — it pre-installs the
full toolchain and hides regressions. Report pip status, compiled yes/no,
import status, capture shape, and the last lines of any failing log;
don't paper over a missing system dep by installing it.

### Triage a "pip install failed" issue

Extract distro + version, Python version, exact command, traceback, and
venv vs system pip. Reproduce in a container matching the distro with
only what the reporter had. Typical root causes: missing `libx11-dev`/
`gcc`; Python ABI mismatch on a stale `.so`; resolver picking an
incompatible numpy. Windows/macOS reports can't be reproduced in Linux
containers — say so.

## 10. Review checklist (apply to every PR)

- [ ] Default wheel still has `numpy` as its only runtime dep.
- [ ] New optional functionality is behind an extra with lazy imports and
      an install hint.
- [ ] BGRA contract preserved; buffer reuse preserved.
- [ ] `docker compose run --rm test` passes; wayland suite run if backends
      touched.
- [ ] `test_screenshot.py` still platform-agnostic.
- [ ] README and `backends/__init__.py` docstring updated for user-visible
      changes; `poetry.lock` regenerated if `pyproject.toml` deps changed.
- [ ] No host-only assumptions (fonts, display numbers, paths) baked in;
      fonts go through `$FASTGRAB_FONT` / `_find_font()`.
