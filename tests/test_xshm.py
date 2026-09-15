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
        # Not skip(). Any death here -- a crash in the extension, a
        # segfault in the parallel copy, a missing python-xlib -- used
        # to report as "skipped", which reads as green. conftest's
        # require_some_display owns the no-display case, and this module
        # only runs where the C extension imported, so a child that
        # cannot finish is a failure.
        pytest.fail("capture child exited %d: %s"
                    % (proc.returncode, proc.stderr.strip()[-500:]))
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
    # Not asserted here: that shm_used is True. A build without the
    # extension, or a remote DISPLAY, legitimately falls back — and the
    # digests matching is the guarantee that actually matters. Where the
    # fast path *is* guaranteed, the next test says so and asserts it.
    if shm_used:
        assert shm_count >= 1


def test_the_shm_path_really_runs_where_it_is_guaranteed():
    """A fallback nobody notices is the whole risk of this change.

    Every other test in this file passes identically on either path —
    which is the point, and also why none of them would notice the shm
    path disappearing altogether. A typo in the XShmQueryExtension
    check, or issue #72's stale .so being imported instead of the
    freshly built one, would put every capture back on XGetImage while
    CI stayed green and the README went on advertising an order of
    magnitude it no longer delivered.

    So where shm cannot legitimately fail — a local Xvfb inside this
    project's own image, sharing the IPC namespace — the compose service
    says so with FASTGRAB_EXPECT_XSHM=1, and this asserts it engaged.
    """
    if os.environ.get("FASTGRAB_EXPECT_XSHM") != "1":
        pytest.skip(
            "shm is not guaranteed on this display; the docker compose "
            "'test' service sets FASTGRAB_EXPECT_XSHM=1 where it is"
        )
    _, shm_used, shm_count = _child({})
    assert shm_used is True, (
        "the shm path did not engage on a display where it must: the "
        "extension probe, the attach, or the build has regressed, and "
        "every capture is silently taking the slow XGetImage path"
    )
    assert shm_count >= 1


def test_the_shipped_hot_path_matches_the_serial_copy():
    """shm *and* the parallel copy — what every >=1080p capture runs.

    The threshold test below forces FASTGRAB_NO_XSHM=1 on both children,
    and the default child captures 640x480, which is under the 4 MiB
    threshold and so copies serially. That left the one combination
    users actually get with no coverage at all: a chunking bug that only
    appears when the OpenMP loop reads out of the shm segment would pass
    the entire suite.
    """
    serial, _, _ = _child({"FASTGRAB_OMP_MIN_BYTES": "0"})
    parallel, _, _ = _child({"FASTGRAB_OMP_MIN_BYTES": "1"})
    assert serial == parallel, (
        "the parallel copy out of the shm segment disagrees with the "
        "serial one: %s vs %s" % (serial, parallel)
    )


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
