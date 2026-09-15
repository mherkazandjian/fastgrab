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


def _run(script, env_overrides=None):
    """Run a snippet in a fresh interpreter and hand back the result."""
    env = dict(os.environ)
    env.update(env_overrides or {})
    env.setdefault("PYTHONPATH", os.pathsep.join(sys.path))
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, env=env, timeout=180)


# Warm the parent with a parallel region, fork, and have the child copy
# several large frames. Before the guard this child blocked every time.
_FORK_SCRIPT = r"""
import os
import select

from fastgrab import screenshot, _linux_x11

# The caller forces $FASTGRAB_OMP_MIN_BYTES to 1, so every capture here
# takes the parallel branch whatever this display measures. Requiring a
# large screen instead -- which is what this did first -- made the test
# fail outright on an ordinary 1280x720 or 1024x768, 3.5 MB and 3.0 MB,
# both under the 4 MiB default. That is a bug in the test, not in
# capture, and it would have found its way onto any developer whose
# desktop is smaller than the CI container's 1280x1024.

grab = screenshot.Screenshot()
grab.capture()
parent = _linux_x11._display_cache_info()
if parent["forked"] or not parent["omp_allowed"]:
    print("SETUP-FAILED: the parent is not the importing process: %r" % parent)
    raise SystemExit(2)
if parent["omp_parallel_copies"] < 1:
    print("SETUP-FAILED: the parent never took the parallel copy, so the "
          "libgomp pool was never warmed and the child proves nothing: %r"
          % parent)
    raise SystemExit(2)

read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    try:
        before = _linux_x11._display_cache_info()
        child = screenshot.Screenshot()
        for _ in range(3):
            child.capture()
        after = _linux_x11._display_cache_info()
        os.write(write_fd, ("OK %s %s %d" % (
            after["forked"], after["omp_allowed"],
            after["omp_parallel_copies"] - before["omp_parallel_copies"]
        )).encode())
    except BaseException as exc:
        os.write(write_fd, ("ERR %r" % (exc,)).encode())
    finally:
        os._exit(0)

os.close(write_fd)
ready, _w, _x = select.select([read_fd], [], [], 30)
if not ready:
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    print("CHILD-HUNG: the forked child never finished its captures")
    raise SystemExit(3)
message = os.read(read_fd, 4096).decode()
os.waitpid(pid, 0)

# The parent has to outlive its child, not merely start it.
grab.capture()
print(message)
"""


# The override only has to flip the decision. Deliberately no large
# capture here: it re-arms a path measured to deadlock, and a test must
# never depend on a deadlock failing to happen.
_FORK_OVERRIDE_SCRIPT = r"""
import os

from fastgrab import screenshot, _linux_x11

grab = screenshot.Screenshot()
grab.capture()

read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    try:
        info = _linux_x11._display_cache_info()
        os.write(write_fd, ("%s %s" % (info["forked"],
                                       info["omp_allowed"])).encode())
    except BaseException as exc:
        os.write(write_fd, ("ERR %r" % (exc,)).encode())
    finally:
        os._exit(0)

os.close(write_fd)
message = os.read(read_fd, 4096).decode()
os.waitpid(pid, 0)
print(message)
"""


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


def _omp_compiled():
    return bool(_linux_x11._display_cache_info()["omp_compiled"])


def test_openmp_survived_the_build_where_it_must_have():
    """build.py probes for OpenMP now, and a probe can fall back silently.

    -fopenmp went missing once already: build.py linked gomp from the
    day the extension was written but never asked the compiler to honour
    the pragmas, so every parallel region was quietly ignored for the
    life of the project and nothing noticed. Falling back to a serial
    build rather than failing the install is the right call for a package
    compiled on the user's machine -- and it recreates that exact trap
    unless somebody asserts the flag survived where it had no excuse not
    to.
    """
    if os.environ.get("FASTGRAB_EXPECT_OPENMP") != "1":
        pytest.skip(
            "OpenMP is not guaranteed for this build; the docker compose "
            "'test' service sets FASTGRAB_EXPECT_OPENMP=1 where it is"
        )
    assert _omp_compiled() is True, (
        "the extension was built without OpenMP: build.py's probe fell "
        "back, so every large frame is copied on one core"
    )


_PARALLEL_COPY_SCRIPT = r"""
from fastgrab import screenshot, _linux_x11

before = _linux_x11._display_cache_info()["omp_parallel_copies"]
screenshot.Screenshot().capture()
info = _linux_x11._display_cache_info()
print("%s %d" % (info["omp_compiled"],
                 info["omp_parallel_copies"] - before))
"""


def test_a_capture_really_copies_in_parallel():
    """And that the flag surviving actually reaches the copy.

    omp_compiled says the pragmas were compiled in; this says one of
    them ran. The counter is incremented inside the same #ifdef as the
    pragma precisely so that it cannot answer for a serial build -- it
    used to sit outside, and a build with no OpenMP runtime loaded at all
    reported 201 parallel copies on a single thread.

    Run in a child with the threshold forced to one byte rather than
    against whatever this display happens to measure: keying it on the
    screen size would skip silently on a 1280x720 desktop, which is
    where a vacuous pass comes from.
    """
    if os.environ.get("FASTGRAB_EXPECT_OPENMP") != "1":
        pytest.skip("OpenMP is not guaranteed for this build")
    proc = _run(_PARALLEL_COPY_SCRIPT, {"FASTGRAB_OMP_MIN_BYTES": "1"})
    assert proc.returncode == 0, proc.stderr.strip()[-400:]
    compiled, parallel_copies = proc.stdout.strip().split()
    assert compiled == "True", "the extension was built without OpenMP"
    assert int(parallel_copies) >= 1, (
        "the capture took the serial copy even with the threshold at one "
        "byte and OpenMP compiled into the extension"
    )


def test_a_forked_child_does_not_deadlock_on_a_large_capture():
    """libgomp is not fork-safe, and this project supports fork-and-capture.

    Once a process has run a parallel region its libgomp pool is kept for
    reuse; fork() copies the calling thread's bookkeeping into the child
    without the pool's threads, and the child's next region waits at a
    barrier for workers that no longer exist. Measured against this code
    before the guard: the child blocked every run, in futex_wait inside
    gomp_team_barrier_wait_end under fg_copy_image. The existing
    test_a_forked_child_connects_for_itself cannot see it -- that child
    only calls resolution(), so it never enters a region.

    Finishing is not enough to assert. A guard that quietly stopped
    working would also pass on any run where the deadlock happened not to
    reproduce, so the child has to show it went serial: forked, OpenMP
    disarmed, and no copy took the parallel branch.
    """
    if not _omp_compiled():
        pytest.skip(
            "built without OpenMP, so there is no libgomp pool to inherit "
            "and the hazard this pins cannot arise"
        )
    proc = _run(_FORK_SCRIPT, {"FASTGRAB_OMP_MIN_BYTES": "1"})
    assert proc.returncode == 0, (
        "the forked child did not come back: %s %s"
        % (proc.stdout.strip(), proc.stderr.strip()[-400:])
    )
    status, forked, allowed, parallel_copies = proc.stdout.strip().split()
    assert status == "OK", proc.stdout.strip()
    assert forked == "True", "the child did not recognise itself as forked"
    assert allowed == "False", "the guard left OpenMP armed in the child"
    assert int(parallel_copies) == 0, (
        "the child took the parallel copy %s times despite the guard"
        % parallel_copies
    )


@pytest.mark.parametrize("value,allowed", [
    ("1", "True"), ("yes", "True"), ("0", "False"), ("", "False"),
])
def test_the_fork_guard_can_be_turned_off(value, allowed):
    """$FASTGRAB_UNSAFE_OMP_AFTER_FORK, for callers who accept the risk.

    The guard is conservative by construction: it disarms OpenMP in every
    inherited process, including the ones that forked from a runtime
    which never warmed a pool and would have been perfectly safe. This is
    the way out for someone who knows that about their own program.

    Only the decision is asserted, never a capture -- the override
    re-arms a path measured to deadlock, and a test that took a large
    frame with it on would be betting on a hang not happening.
    """
    proc = _run(_FORK_OVERRIDE_SCRIPT,
                {"FASTGRAB_UNSAFE_OMP_AFTER_FORK": value})
    assert proc.returncode == 0, proc.stderr.strip()[-400:]
    forked, is_allowed = proc.stdout.strip().split()
    assert forked == "True"
    assert is_allowed == allowed, (
        "%r should have left the guard %s"
        % (value, "off" if allowed == "True" else "on")
    )
