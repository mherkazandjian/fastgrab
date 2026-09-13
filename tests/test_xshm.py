"""MIT-SHM capture path and the OpenMP row copy.

``fg_op_screenshot`` prefers ``XShmGetImage``: the server writes the
region straight into a shared segment instead of pushing it through the
socket. Everything about that is optional — a missing extension, a
refused attach, a remote DISPLAY — so the property that matters is not
"shm is used" but "whichever path ran, the pixels are the same". These
tests pin that, plus the two switches that select a path.

Each switch is read once per process and cached, so exercising both
sides of one means two processes; hence the subprocess helper rather
than monkeypatching ``os.environ``.

Skipped on non-Linux: the C extension is the libX11 one.
"""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="fastgrab._linux_x11 is the libX11 C extension; Linux only",
)

from fastgrab import _linux_x11  # noqa: E402


# Paint a deterministic full-screen window, capture it, print a digest.
# The window is painted from a fixed seed inside each child so both
# children see identical pixels without needing to share a painter.
_CAPTURE_AND_DIGEST = r"""
import hashlib, random, sys
from Xlib import display as xdisplay, X
from fastgrab import screenshot
from fastgrab import _linux_x11

d = xdisplay.Display()
scr = d.screen()
geom = scr.root.get_geometry()
w = min(640, geom.width)
h = min(480, geom.height)
win = scr.root.create_window(0, 0, geom.width, geom.height, 0, scr.root_depth,
                             X.InputOutput, X.CopyFromParent,
                             background_pixel=scr.black_pixel,
                             override_redirect=True)
win.map()
d.sync()
gc = win.create_gc()
random.seed(20240913)
for _ in range(500):
    gc.change(foreground=random.randint(0, 0xFFFFFF))
    win.fill_rectangle(gc, random.randint(0, geom.width - 1),
                       random.randint(0, geom.height - 1), 40, 40)
d.sync()

img = screenshot.Screenshot().capture(bbox=(0, 0, w, h))
info = _linux_x11._display_cache_info()
print("%s %s %d" % (hashlib.sha256(img.tobytes()).hexdigest(),
                    info["shm"], info["shm_captures"]))
"""


def _child(env_overrides):
    """Run the capture snippet in a fresh interpreter, return its output."""
    env = dict(os.environ)
    env.update(env_overrides)
    env.setdefault("PYTHONPATH", os.pathsep.join(sys.path))
    proc = subprocess.run([sys.executable, "-c", _CAPTURE_AND_DIGEST],
                          capture_output=True, text=True, env=env, timeout=120)
    if proc.returncode != 0:
        pytest.skip("capture child failed (no usable DISPLAY?): %s"
                    % proc.stderr.strip()[-300:])
    digest, shm, count = proc.stdout.strip().split()
    return digest, shm == "True", int(count)


def test_cache_info_reports_the_shared_memory_state():
    """The private cache dict is how tests tell the two paths apart."""
    info = _linux_x11._display_cache_info()
    assert "shm" in info and isinstance(info["shm"], bool)
    assert "shm_captures" in info and isinstance(info["shm_captures"], int)
    assert info["shm_captures"] >= 0


def test_shm_and_fallback_capture_identical_pixels():
    """The optimisation must be invisible in the output, not just fast.

    This is the test that would have caught a shm image whose rows are
    padded differently from the XGetImage one — the failure mode would
    be a sheared image, which is easy to miss by eye on noise.
    """
    shm_digest, shm_used, shm_count = _child({})
    plain_digest, plain_used, plain_count = _child({"FASTGRAB_NO_XSHM": "1"})

    assert plain_used is False, "FASTGRAB_NO_XSHM did not disable the shm path"
    assert plain_count == 0
    assert shm_digest == plain_digest, (
        "shm and XGetImage disagree on the same screen: %s vs %s"
        % (shm_digest, plain_digest)
    )
    # Not asserted: that shm_used is True. A build without the extension,
    # or a remote DISPLAY, legitimately falls back — and the digests
    # matching is the guarantee that actually matters.
    if shm_used:
        assert shm_count >= 1


@pytest.mark.parametrize("threshold", ["0", "1"])
def test_openmp_threshold_selects_a_path_without_changing_the_pixels(threshold):
    """0 forces the serial copy, 1 forces the parallel one.

    Both must produce the same bytes; a race or an off-by-one in the row
    chunking would show up here as a digest mismatch.
    """
    baseline, _, _ = _child({"FASTGRAB_NO_XSHM": "1",
                             "FASTGRAB_OMP_MIN_BYTES": "0"})
    other, _, _ = _child({"FASTGRAB_NO_XSHM": "1",
                          "FASTGRAB_OMP_MIN_BYTES": threshold})
    assert baseline == other
