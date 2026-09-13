import os
import shutil
import socket
import subprocess
import sys
import time

import pytest


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
    """Whether an X display can actually be opened.

    Asks the extension the tests themselves will use, rather than
    shelling out to xdpyinfo or reimplementing Xlib's address parsing.
    Xlib accepts transport prefixes, socket pathnames, bracketed IPv6
    literals and address-family pins; anything that reinterprets DISPLAY
    by hand gets some of those wrong and mis-gates the suite. The old
    xdpyinfo check also degraded to a bare "DISPLAY is set" whenever
    xdpyinfo was not installed, which is the hole behind #44.
    """
    if not os.environ.get("DISPLAY"):
        return False
    try:
        from fastgrab import _linux_x11
    except ImportError:
        # No compiled extension means nothing here can capture anyway.
        return False
    try:
        _linux_x11.resolution()
    except Exception:
        return False
    return True


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
        # "blocks" is the painter's coordinate-encoding pattern; anything
        # else is a solid #RRGGBB. A flat color cannot distinguish a
        # correctly placed sub-region capture from a displaced one, which
        # is what the scale tests need to check.
        command = "blocks" if color == "blocks" else "paint " + color
        cs.sendall((command + "\n").encode())
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


def _wlr_randr(*args):
    """Run ``wlr-randr`` against the current Wayland display."""
    return subprocess.run(
        ["wlr-randr", *args],
        check=True, capture_output=True, text=True,
    )


def _wlr_first_output_name():
    """The name of the first output ``wlr-randr`` lists (cage has one)."""
    for line in _wlr_randr().stdout.splitlines():
        if line and not line[0].isspace():
            return line.split()[0]
    return None


def _refresh_wlr_output_state():
    """Round-trip the shared wlr connection so latched geometry updates.

    ``WlrBackend`` keeps one connection and one set of ``_OutputState``
    objects per process, and reads them without querying the compositor.
    A scale change is therefore invisible — to every ``Screenshot``
    already built *and* to every one built afterwards — until something
    dispatches. Doing it here rather than in each test keeps a test that
    changed the scale from leaving a stale reading behind for the next
    one.
    """
    from fastgrab.backends.wlr import WlrBackend

    WlrBackend().refresh()


@pytest.fixture
def wlr_output_scale():
    """Set the wlroots output scale for the duration of one test.

    cage's headless output comes up at scale 1 and neither cage nor the
    wlroots headless backend takes a scale option or environment
    variable. cage does implement ``zwlr_output_manager_v1`` though, so
    ``wlr-randr`` can drive the scale live — which is the only way to
    reach the scaled sub-region path of the wlr backend under
    ``docker compose run --rm test-wayland``.

    Yields ``set_scale(2)``. The scale is put back to 1 on teardown,
    and the backend's latched output state is refreshed both times so
    neither this test nor the next one reads a stale logical size.
    """
    if not _wayland_available():
        pytest.skip("wlr_output_scale requires a working Wayland display")
    if not shutil.which("wlr-randr"):
        pytest.skip("wlr-randr is not installed (apt install wlr-randr)")

    name = _wlr_first_output_name()
    if not name:
        pytest.skip("wlr-randr listed no outputs")

    def _set_scale(scale) -> None:
        _wlr_randr("--output", name, "--scale", str(scale))
        # The compositor reconfigures the kiosk client, which redraws at
        # the new size; give both a moment before anything is captured.
        time.sleep(0.3)
        _refresh_wlr_output_state()

    yield _set_scale

    try:
        _wlr_randr("--output", name, "--scale", "1")
        time.sleep(0.3)
        _refresh_wlr_output_state()
    except Exception:
        pass
