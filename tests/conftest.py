import os
import shutil
import subprocess

import pytest


def _display_available() -> bool:
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    if not shutil.which("xdpyinfo"):
        return True
    return subprocess.run(
        ["xdpyinfo", "-display", display],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


@pytest.fixture(scope="session", autouse=True)
def require_display():
    if not _display_available():
        pytest.skip(
            "no working X11 DISPLAY (run via docker compose, xvfb-run, or "
            "an existing X session)",
            allow_module_level=True,
        )


@pytest.fixture
def paint_root():
    """Paint the X root window a solid color via XFillRectangle.

    Yields a callable: ``paint("#RRGGBB")``. ``xsetroot -solid`` only sets the
    background pixmap, which is invisible to ``XGetImage`` on a bare Xvfb
    screen with no compositor — so we draw directly into the root drawable.
    """
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
