import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

from fastgrab.backends._display import probe_display


SOCKET_NAME = "fastgrab-painter.sock"


# Pytest looks at this module-level list and refuses to *import* the named
# files during collection. Marker filters (-m "not wayland") and
# pytestmark = pytest.mark.skipif(...) only skip test functions — they
# don't prevent the test module's imports from running, which is where
# the cross-platform CI jobs (windows, macos) tripped: test_integration_wlr.py
# imports pywayland, test_x11_lowlevel.py imports the libX11 C extension.
collect_ignore = []
if sys.platform != "linux":
    collect_ignore.extend([
        "test_integration_wlr.py",   # imports fastgrab.backends.wlr → pywayland
        "test_x11_lowlevel.py",      # imports fastgrab._linux_x11
    ])


def _x11_available() -> bool:
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    if not shutil.which("xdpyinfo"):
        # Probe the server socket rather than trusting the variable.
        # Returning True here waved the display tests through against a
        # server that was not listening, turning what should be a clean
        # skip into a "cannot open X display" failure — issue #44.
        return probe_display(display)
    return subprocess.run(
        ["xdpyinfo", "-display", display],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _wayland_available() -> bool:
    wd = os.environ.get("WAYLAND_DISPLAY")
    rd = os.environ.get("XDG_RUNTIME_DIR")
    if not wd or not rd:
        return False
    return os.path.exists(os.path.join(rd, wd))


_SOME_DISPLAY = None


def _some_display_available() -> bool:
    # Cached: the gate is consulted once per test, and _x11_available()
    # shells out to xdpyinfo.
    global _SOME_DISPLAY
    if _SOME_DISPLAY is None:
        _SOME_DISPLAY = _x11_available() or _wayland_available()
    return _SOME_DISPLAY


@pytest.fixture(autouse=True)
def require_some_display(request):
    """Skip tests that need a display when there is not one.

    Per-test rather than per-session, and skippable with the
    ``no_display`` marker: suites that drive fakes (the macOS and wlr
    backend unit tests) need no display at all, and a session-wide skip
    took them down with everything else — losing exactly the coverage a
    contributor gets when running pytest on a headless box, which is
    also the only place ``off_x``/``off_y`` are ever exercised.
    """
    # On Windows/macOS the system always has a desktop available to capture;
    # the display-server gate is a Linux-only concept.
    if sys.platform != "linux":
        return
    if request.node.get_closest_marker("no_display"):
        return
    if not _some_display_available():
        pytest.skip("no usable display server (need DISPLAY or WAYLAND_DISPLAY)")


@pytest.fixture
def paint_root():
    """Paint the X root window a solid color via XFillRectangle.

    Yields a callable: ``paint("#RRGGBB")``. ``xsetroot -solid`` only sets the
    background pixmap, which is invisible to ``XGetImage`` on a bare Xvfb
    screen with no compositor — so we draw directly into the root drawable.
    """
    if not _x11_available():
        pytest.skip("paint_root requires a working X11 DISPLAY")
    try:
        from Xlib import display as xdisplay
    except ImportError:
        pytest.skip("python-xlib not installed (pip install python-xlib)")

    d = xdisplay.Display()
    screen = d.screen()
    root = screen.root
    width, height = screen.width_in_pixels, screen.height_in_pixels

    def _paint(color: str) -> None:
        if not color.startswith("#") or len(color) != 7:
            raise ValueError(f"expected #RRGGBB, got {color!r}")
        pixel = int(color[1:], 16)
        gc = root.create_gc(foreground=pixel, background=0)
        root.fill_rectangle(gc, 0, 0, width, height)
        gc.free()
        d.sync()

    yield _paint

    d.close()


@pytest.fixture
def paint_wayland():
    """Drive the test-only wayland_painter via its UNIX socket.

    The painter is the kiosk client of cage in the ``test-wayland`` docker
    compose service. It listens on
    ``$XDG_RUNTIME_DIR/fastgrab-painter.sock``. Yields a callable
    ``paint("#RRGGBB")`` that sends a paint command and waits for ack.
    """
    if not _wayland_available():
        pytest.skip("paint_wayland requires a working Wayland display")
    runtime = os.environ["XDG_RUNTIME_DIR"]
    sock_path = os.path.join(runtime, SOCKET_NAME)
    if not os.path.exists(sock_path):
        pytest.skip(
            "fastgrab-painter socket missing at {} — was cage started via "
            "start-wayland-desktop?".format(sock_path)
        )

    cs = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cs.connect(sock_path)
    cs.settimeout(5.0)

    def _paint(color: str) -> None:
        cs.sendall(("paint " + color + "\n").encode())
        # Read until newline (the painter responds 'ok\n' or 'err: ...\n').
        buf = b""
        while b"\n" not in buf:
            chunk = cs.recv(64)
            if not chunk:
                raise RuntimeError("painter closed the connection")
            buf += chunk
        line = buf.split(b"\n", 1)[0].decode()
        if not line.startswith("ok"):
            raise RuntimeError("painter error: " + line)
        # Give the compositor a brief moment to render the new commit
        # before the test calls capture(). Without this, the first
        # capture after a paint can race with the swap.
        time.sleep(0.05)

    yield _paint

    try:
        cs.close()
    except Exception:
        pass
