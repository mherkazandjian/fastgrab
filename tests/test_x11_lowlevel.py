"""Direct exercises of the libX11 C extension.

These tests bypass the Screenshot dispatcher and call into
``fastgrab._linux_x11`` directly. They lock down the wire-level
contract (shape, BGRA byte order, resolution-equality with the
high-level wrapper) that the X11Backend depends on.

Skipped on non-Linux: the C extension is not built into wheels for
Windows or macOS.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="fastgrab._linux_x11 is the libX11 C extension; Linux only",
)

from fastgrab import screenshot  # noqa: E402
from fastgrab import _linux_x11  # noqa: E402


def test_low_level_resolution_returns_positive_2_tuple():
    res = _linux_x11.resolution()
    assert isinstance(res, tuple)
    assert len(res) == 2
    width, height = res
    assert isinstance(width, int) and isinstance(height, int)
    assert width > 0 and height > 0


def test_low_level_bytes_per_pixel_is_4():
    # X11 ZPixmap on every supported platform we ship to is 32-bit (BGRA).
    assert _linux_x11.bytes_per_pixel() == 4


def test_low_level_screenshot_fills_buffer():
    width, height = _linux_x11.resolution()
    buf = numpy.zeros((height, width, 4), dtype="uint8")
    _linux_x11.screenshot(0, 0, buf)
    assert buf.shape == (height, width, 4)
    assert buf.dtype == numpy.uint8


def test_screensize_matches_low_level_resolution():
    grab = screenshot.Screenshot(backend="x11")
    assert grab.screensize == _linux_x11.resolution()


def test_unreachable_display_raises_instead_of_segfaulting(monkeypatch):
    # Regression: every entry point used to dereference a NULL Display*
    # when XOpenDisplay failed, crashing the interpreter (exit 139).
    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.resolution()
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.bytes_per_pixel()
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.screenshot(0, 0, numpy.zeros((2, 2, 4), dtype=numpy.uint8))
    # And through the public API — Screenshot() itself doesn't touch the
    # server, the first capture() does.
    from fastgrab import screenshot
    with pytest.raises(RuntimeError, match="cannot open X display"):
        screenshot.Screenshot(backend="x11").capture()


@pytest.mark.parametrize("x, y", [(-1, -1), (-1, 0), (0, -1)])
def test_low_level_screenshot_rejects_negative_origin(x, y):
    """Regression: a negative origin used to kill the interpreter.

    It reaches XGetImage as a BadMatch, and X protocol errors are
    delivered to Xlib's default error handler, which prints a
    diagnostic and calls exit() — no exception, no traceback, and the
    NULL check in the C code never gets a chance to run.
    """
    buf = numpy.zeros((8, 8, 4), dtype="uint8")
    with pytest.raises(RuntimeError, match="outside the screen"):
        _linux_x11.screenshot(x, y, buf)


def test_low_level_screenshot_rejects_region_past_the_right_edge():
    width, height = _linux_x11.resolution()
    buf = numpy.zeros((8, 8, 4), dtype="uint8")
    with pytest.raises(RuntimeError, match="outside the screen"):
        _linux_x11.screenshot(width - 4, 0, buf)
    with pytest.raises(RuntimeError, match="outside the screen"):
        _linux_x11.screenshot(0, height - 4, buf)


def test_low_level_screenshot_rejects_empty_region():
    with pytest.raises(ValueError, match="must be positive"):
        _linux_x11.screenshot(0, 0, numpy.zeros((0, 8, 4), dtype="uint8"))


def test_screenshot_rejects_non_3d_buffer():
    with pytest.raises(ValueError, match="height, width, 4"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8), dtype=numpy.uint8))


def test_screenshot_rejects_buffer_without_four_channels():
    """Regression: this used to overrun the destination buffer.

    Only ``ndim == 3`` was checked, so an (8, 8, 3) array was accepted
    and the 32-bpp copy wrote 8*8*4 = 256 bytes into a 192-byte
    allocation — a heap overflow through the advertised low-level API.
    """
    with pytest.raises(ValueError, match="height, width, 4"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8, 3), dtype=numpy.uint8))


def test_screenshot_rejects_wrong_dtype():
    """Rejected, not coerced: a coerced copy would be filled and dropped."""
    with pytest.raises(ValueError, match="dtype uint8"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8, 4), dtype=numpy.float64))


def test_screenshot_rejects_non_contiguous_buffer():
    """A strided view cannot be filled in place, so it must not be taken."""
    view = numpy.zeros((8, 16, 4), dtype=numpy.uint8)[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]  # sanity: the case under test
    with pytest.raises(ValueError, match="C-contiguous"):
        _linux_x11.screenshot(0, 0, view)


def test_screenshot_rejects_read_only_buffer():
    buf = numpy.zeros((8, 8, 4), dtype=numpy.uint8)
    buf.flags.writeable = False
    with pytest.raises(ValueError, match="C-contiguous|writable"):
        _linux_x11.screenshot(0, 0, buf)


def test_screenshot_rejects_non_array_buffer():
    with pytest.raises(TypeError, match="ndarray"):
        _linux_x11.screenshot(0, 0, [[0, 0, 0, 0]])


def test_display_failure_names_the_display_it_tried(monkeypatch):
    # The extension reports every XOpenDisplay failure with one sentence,
    # so an intermittent CI failure used to be uninvestigable: the log
    # could not say whether DISPLAY was unset, pointed somewhere dead, or
    # had simply been refused that once. Regression for issue #44.
    from fastgrab.backends.x11 import X11Backend

    monkeypatch.setenv("DISPLAY", ":77")
    backend = X11Backend()
    with pytest.raises(RuntimeError, match=r"cannot open X display.*DISPLAY=':77'"):
        backend.resolution()
    with pytest.raises(RuntimeError, match="still unreachable on retry"):
        backend.screenshot(0, 0, numpy.zeros((2, 2, 4), dtype=numpy.uint8))

    # The original error stays reachable as the cause rather than being
    # swallowed and reworded.
    try:
        backend.resolution()
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, RuntimeError)
        assert "cannot open X display" in str(exc.__cause__)


def test_bytes_per_pixel_also_reports_the_display(monkeypatch):
    # It opens a display of its own, and once Screenshot has cached the
    # screen size it is the first server call a later capture() makes —
    # so it is the likeliest entry point to meet a vanished server.
    from fastgrab.backends.x11 import X11Backend

    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match=r"DISPLAY=':77'"):
        X11Backend().bytes_per_pixel()


def test_unset_and_empty_display_are_distinguished(monkeypatch):
    from fastgrab.backends.x11 import _describe_display_state

    monkeypatch.delenv("DISPLAY", raising=False)
    assert _describe_display_state() == "DISPLAY is not set"
    monkeypatch.setenv("DISPLAY", "")
    assert _describe_display_state() == "DISPLAY is set but empty"


def test_a_transient_refusal_is_called_out(monkeypatch):
    # The discriminator #44 needs: if the very next connection succeeds,
    # the server never went away and the failure was transient.
    import fastgrab.backends.x11 as x11_mod

    class _Works:
        @staticmethod
        def resolution():
            return (1920, 1080)

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(x11_mod, "_linux_x11", _Works)
    assert "reachable on retry" in x11_mod._describe_display_state()


def test_a_persistent_outage_is_called_out(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":77")
    from fastgrab.backends.x11 import _describe_display_state

    assert "still unreachable on retry" in _describe_display_state()


def test_non_display_errors_pass_through_unchanged():
    # Only the open-display error is annotated. Driven through the
    # decorator directly because the extension's other RuntimeErrors
    # (a failed XGetImage) cannot be provoked from a healthy server.
    from fastgrab.backends.x11 import _with_display_context

    @_with_display_context
    def boom():
        raise RuntimeError("XGetImage returned NULL")

    with pytest.raises(RuntimeError) as excinfo:
        boom()
    assert "DISPLAY=" not in str(excinfo.value)
    assert str(excinfo.value) == "XGetImage returned NULL"


def test_a_failing_diagnostic_does_not_replace_the_real_error(monkeypatch):
    # Callers catch RuntimeError. If describing DISPLAY blows up, the
    # original error must still be what comes out.
    import fastgrab.backends.x11 as x11_mod

    def boom():
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(x11_mod, "_describe_display_state", boom)
    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match="cannot open X display") as excinfo:
        x11_mod.X11Backend().resolution()
    assert "DISPLAY=" not in str(excinfo.value)


def test_display_context_preserves_the_wrapped_signature():
    from fastgrab.backends.x11 import X11Backend

    assert X11Backend.resolution.__name__ == "resolution"
    assert X11Backend.screenshot.__name__ == "screenshot"
    assert X11Backend.bytes_per_pixel.__name__ == "bytes_per_pixel"


# --------------------------------------------------------------------
# The cached display connection (issue #44)
#
# The extension used to run XOpenDisplay(NULL) ... XCloseDisplay() around
# every single request, so a capture loop opened and tore down one X
# connection per frame. It now keeps one connection per process, keyed on
# DISPLAY. These tests pin the four properties that make that safe:
# connection reuse, honouring a changed DISPLAY, not inheriting the
# connection across fork(), and turning a server that dies underneath a
# cached connection into an exception rather than an exit().
#
# ``_display_cache_info()`` is private and exists for these tests. There
# is no honest way to observe "did that open a second connection?"
# through the public API, and a test that cannot tell reuse from churn
# would not be testing the fix at all.
# --------------------------------------------------------------------


def test_display_cache_info_reports_the_live_connection():
    _linux_x11.resolution()
    info = _linux_x11._display_cache_info()
    assert info["connected"] is True
    assert info["display"] == os.environ["DISPLAY"]
    assert info["pid"] == os.getpid()
    assert info["opens"] >= 1


def test_repeated_captures_reuse_a_single_connection():
    # The fix itself: 150 server requests over one connection. Before the
    # cache that was 150 connects and 150 disconnects, which is the churn
    # #44 blames for the intermittent "cannot open X display".
    _linux_x11.resolution()
    before = _linux_x11._display_cache_info()
    assert before["connected"] is True

    fds_before = len(os.listdir("/proc/self/fd"))
    buf = numpy.zeros((16, 16, 4), "uint8")
    for _ in range(50):
        _linux_x11.screenshot(0, 0, buf)
        _linux_x11.bytes_per_pixel()
        _linux_x11.resolution()

    after = _linux_x11._display_cache_info()
    assert after["opens"] == before["opens"], "a capture loop reconnected"
    assert after["connected"] is True
    # A reused connection must also not leak a descriptor per call: the
    # recovery path drops connections, and dropping one without
    # reclaiming its descriptor would exhaust the process instead of the
    # server.
    assert len(os.listdir("/proc/self/fd")) == fds_before


def test_a_changed_display_is_honoured(monkeypatch):
    # A plain static Display* would keep serving the old connection here,
    # so the capture would quietly succeed against the display the caller
    # just stopped naming. Keyed on DISPLAY it reconnects instead — which
    # is also what keeps
    # test_unreachable_display_raises_instead_of_segfaulting meaningful.
    real = os.environ["DISPLAY"]
    _linux_x11.resolution()
    first = _linux_x11._display_cache_info()
    assert first["display"] == real

    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.resolution()
    dropped = _linux_x11._display_cache_info()
    assert dropped["connected"] is False, "kept a connection to the old display"
    assert dropped["opens"] == first["opens"], "a failed open must not count"

    monkeypatch.setenv("DISPLAY", real)
    _linux_x11.resolution()
    back = _linux_x11._display_cache_info()
    assert back["display"] == real
    assert back["opens"] == first["opens"] + 1


def test_unset_and_empty_display_are_separate_cache_keys(monkeypatch):
    # XOpenDisplay(NULL) treats the two differently, so the cache has to
    # as well: neither may be answered from a connection that was opened
    # while DISPLAY named a real server.
    real = os.environ["DISPLAY"]
    _linux_x11.resolution()
    assert _linux_x11._display_cache_info()["connected"] is True

    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.resolution()
    assert _linux_x11._display_cache_info()["connected"] is False

    monkeypatch.setenv("DISPLAY", real)
    _linux_x11.resolution()
    assert _linux_x11._display_cache_info()["connected"] is True

    monkeypatch.setenv("DISPLAY", "")
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.resolution()
    assert _linux_x11._display_cache_info()["connected"] is False

    monkeypatch.setenv("DISPLAY", real)
    _linux_x11.resolution()


def test_close_display_drops_the_cache_and_the_next_call_reconnects():
    _linux_x11.resolution()
    before = _linux_x11._display_cache_info()

    _linux_x11._close_display()
    closed = _linux_x11._display_cache_info()
    assert closed["connected"] is False
    assert closed["display"] is None

    _linux_x11.resolution()
    after = _linux_x11._display_cache_info()
    assert after["connected"] is True
    assert after["opens"] == before["opens"] + 1


def test_a_forked_child_connects_for_itself():
    """A child must not speak on the connection it inherited.

    Parent and child hold descriptors onto one socket, and their
    requests interleave into a single protocol stream — the classic way
    a multiprocessing pool corrupts an X connection. The child has to
    notice the pid changed and connect for itself, and it must abandon
    the inherited Display *without* XCloseDisplay(), which would write to
    the socket the parent is still using.
    """
    _linux_x11.resolution()
    parent = _linux_x11._display_cache_info()
    assert parent["connected"] is True

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        # Child. os._exit throughout: running pytest's teardown a second
        # time, or flushing the parent's captured output, is its own mess.
        status = 1
        try:
            os.close(read_fd)
            _linux_x11.resolution()
            info = _linux_x11._display_cache_info()
            os.write(write_fd, repr(info).encode())
            if (info["connected"] is True
                    and info["pid"] == os.getpid()
                    and info["pid"] != parent["pid"]
                    and info["opens"] == parent["opens"] + 1):
                status = 0
        except BaseException as exc:      # reported through the exit status
            try:
                os.write(write_fd, repr(exc).encode())
            except BaseException:
                pass
        finally:
            os._exit(status)

    os.close(write_fd)
    reported = os.read(read_fd, 8192).decode()
    os.close(read_fd)
    _, wait_status = os.waitpid(pid, 0)
    assert os.WIFEXITED(wait_status), "the child died on a signal: " + reported
    assert os.WEXITSTATUS(wait_status) == 0, "child reported: " + reported

    # And the parent's own connection came through untouched — neither
    # closed by the child nor left with a corrupted request stream.
    after = _linux_x11._display_cache_info()
    assert after["opens"] == parent["opens"]
    assert after["pid"] == parent["pid"]
    _linux_x11.resolution()


def _display_number_in_use(number):
    # Both artefacts matter: Xvfb unlinks its socket *and* its lock file
    # on a clean exit, and leaves both behind when it is SIGKILLed.
    return (os.path.exists("/tmp/.X11-unix/X%d" % number)
            or os.path.exists("/tmp/.X%d-lock" % number))


def _free_display_numbers(count):
    numbers = []
    for number in range(90, 130):
        if _display_number_in_use(number):
            continue
        numbers.append(number)
        if len(numbers) == count:
            return numbers
    return None


def _displays_still_in_use(numbers, timeout=15.0):
    """Bounded wait for the disposable servers to release their numbers."""
    deadline = time.time() + timeout
    while True:
        still = [n for n in numbers if _display_number_in_use(n)]
        if not still or time.time() > deadline:
            return still
        time.sleep(0.1)


def _run_x_fault_script(script, displays=1, timeout=180):
    """Run a fault-injection script out of process, return (result, report).

    Out of process because every failure these guard against is the
    interpreter *exiting*: in process it would take the pytest session
    down with it, and an assertion that cannot tell exit-1 from a raised
    exception is not a test of this at all. Here a regression shows up
    as a non-zero exit status plus Xlib's "XIO: fatal IO error" (or "X
    connection to ... broken") on stderr, which the report prints.
    """
    if not shutil.which("Xvfb"):
        pytest.skip("Xvfb is not installed; cannot run a disposable X server")
    numbers = _free_display_numbers(displays)
    if numbers is None:
        pytest.skip("not enough free X display numbers for disposable servers")

    result = subprocess.run(
        [sys.executable, "-c", script] + [":%d" % n for n in numbers],
        capture_output=True, text=True, timeout=timeout,
    )
    leaked = _displays_still_in_use(numbers)
    report = "exit={}\nstdout:\n{}\nstderr:\n{}\nleaked displays: {}".format(
        result.returncode, result.stdout, result.stderr, leaked,
    )
    # A surviving X server fails nothing by itself -- it just keeps a
    # display number. Once the pool is used up these tests start
    # skipping, and a skipped test reads as green, which is the exact
    # failure mode this whole change exists to remove. Fail loudly
    # instead. The report carries the run's own output either way, so
    # this assertion firing first never hides the real problem.
    assert not leaked, report
    return result, report


# Shared by every fault-injection script below. Each one starts throwaway
# X servers, and the three ways that goes wrong are all lifecycle bugs
# rather than bugs in what is being tested, so they are solved once here:
#
#   * a server must be waited for before anything connects to it, or a
#     one-shot XOpenDisplay races Xvfb's startup and fails against a
#     perfectly correct extension;
#   * a server must be SIGTERMed and reaped, not SIGKILLed, or it cannot
#     unlink its socket and lock file and the display number looks
#     occupied to every later run;
#   * every exit path must clean up -- including os._exit() from inside
#     an X I/O error handler, which cannot return and so cannot rely on
#     a finally block.
_XVFB_PRELUDE = r'''
import os
import subprocess
import sys
import time

from fastgrab import _linux_x11

_servers = []


def say(text):
    # os.write rather than print: os._exit() does not flush Python's
    # buffers, and half these scripts exit from inside an X I/O error
    # handler.
    os.write(1, (text + "\n").encode())


def socket_path(display):
    return "/tmp/.X11-unix/X" + display.lstrip(":").split(".")[0]


def start(display):
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "320x240x24", "-ac"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _servers.append(proc)
    return proc


def wait_for_socket(display, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(socket_path(display)):
            return True
        time.sleep(0.05)
    return False


def wait_for_extension(display, timeout=20.0):
    """Bounded wait until the extension can talk to `display`."""
    os.environ["DISPLAY"] = display
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return _linux_x11.resolution()
        except RuntimeError:
            time.sleep(0.05)
    return None


def stop(proc, timeout=15.0):
    """Terminate and reap; SIGKILL only as a bounded fallback.

    SIGTERM lets Xvfb unlink its socket and lock file. SIGKILL does not,
    and the leftovers make that display number look occupied to every
    later run.
    """
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
    else:
        proc.wait()


def cleanup():
    while _servers:
        stop(_servers.pop())


def finish(code, *messages):
    """The one exit path. Nothing may leave a server behind."""
    for message in messages:
        say(message)
    cleanup()
    os._exit(code)
'''


_DEAD_SERVER_SCRIPT = _XVFB_PRELUDE + r'''
import numpy

display = sys.argv[1]
server = start(display)
try:
    if not wait_for_socket(display) or wait_for_extension(display) is None:
        finish(2, "SETUP-FAILED: Xvfb never became reachable")

    buf = numpy.zeros((16, 16, 4), "uint8")
    # Every one of these arms a region and installs the handler. If
    # installing ever recorded the handler as its own predecessor, a
    # fault would recurse instead of longjmping; 50 rounds makes that
    # show up as a crash rather than a pass.
    for _ in range(50):
        _linux_x11.screenshot(0, 0, buf)
    live = _linux_x11._display_cache_info()
    if not (live["connected"] and live["display"] == display):
        finish(2, "SETUP-FAILED: not connected to the private server: %r" % live)

    stop(server)
    time.sleep(0.2)

    # The cached connection is now a dead socket. Xlib's default I/O
    # error handler would print and call exit() right here.
    try:
        _linux_x11.screenshot(0, 0, buf)
    except RuntimeError as exc:
        dead = _linux_x11._display_cache_info()
        if dead["connected"]:
            finish(3, "KEPT-DEAD-CONNECTION: %r" % dead)
        finish(0, "RAISED: %s" % exc, "SURVIVED")
    else:
        finish(3, "NO-ERROR: captured from a server that is gone")
finally:
    cleanup()
'''


def test_a_dead_server_raises_instead_of_exiting_the_interpreter():
    """The hazard the cache introduces, and the reason it is survivable.

    Opening a connection per call meant a server that had gone away was
    met by XOpenDisplay returning NULL — a clean RuntimeError. A cached
    connection meets it as an Xlib I/O error instead, and Xlib's default
    I/O error handler prints a diagnostic and calls exit(): no
    exception, no traceback, the interpreter simply stops. That is the
    same class of failure
    test_low_level_screenshot_rejects_negative_origin describes for
    protocol errors, and trading a flake for a hard exit would be a bad
    bargain.
    """
    result, report = _run_x_fault_script(_DEAD_SERVER_SCRIPT)
    assert result.returncode == 0, report
    assert "SURVIVED" in result.stdout, report
    # Not merely "did not crash": the connection has to have been given
    # up, so the next call reconnects rather than reusing a dead socket.
    assert "cannot open X display" in result.stdout, report


# --------------------------------------------------------------------
# Two ways the cache could still hard-exit the interpreter
#
# Both are the same failure class the I/O error handler exists to
# prevent, and both slipped through the first version of it: it armed
# the guard around the *operation* only, and installed the handler once
# per *connection*. Each test below fault-injects the specific sequence
# and asserts the process survives and raises.
# --------------------------------------------------------------------


_SWITCHED_DISPLAY_SCRIPT = _XVFB_PRELUDE + r'''
dead_display, live_display = sys.argv[1], sys.argv[2]

dead = start(dead_display)
live = start(live_display)
try:
    for display in (dead_display, live_display):
        if not wait_for_socket(display):
            finish(2, "SETUP-FAILED: %s never appeared" % display)

    # Touch the survivor first so both servers are known good, then
    # leave the cache pointing at the one about to be killed.
    if wait_for_extension(live_display) is None:
        finish(2, "SETUP-FAILED: %s never became reachable" % live_display)
    if wait_for_extension(dead_display) is None:
        finish(2, "SETUP-FAILED: %s never became reachable" % dead_display)

    cached = _linux_x11._display_cache_info()
    if cached["display"] != dead_display or not cached["connected"]:
        finish(2, "SETUP-FAILED: cache holds %r" % cached)

    stop(dead)
    time.sleep(0.2)

    # Now name the healthy display. The cache has to drop the stale
    # connection to honour that, and dropping it means XCloseDisplay on
    # a dead socket -- X I/O, and a fatal one if it happens outside an
    # armed region.
    os.environ["DISPLAY"] = live_display
    size = _linux_x11.resolution()
    after = _linux_x11._display_cache_info()
    if not after["connected"] or after["display"] != live_display:
        finish(3, "WRONG-DISPLAY: %r" % after)
    finish(0, "SWITCHED: %r on %s" % (size, after["display"]), "SURVIVED")
finally:
    cleanup()
'''


def test_switching_away_from_a_dead_display_does_not_exit():
    """Dropping a stale connection is itself X I/O.

    The teardown ran outside the armed region, so a server that had died
    while idle made XCloseDisplay fault, the handler fell through to its
    delegation branch, and Xlib's fatal default exited the interpreter --
    even though the display the caller had just switched *to* was
    healthy and the call was about to succeed.
    """
    result, report = _run_x_fault_script(_SWITCHED_DISPLAY_SCRIPT, displays=2)
    assert result.returncode == 0, report
    assert "SURVIVED" in result.stdout, report
    # Survival is not enough: the switch has to have actually landed on
    # the healthy display rather than reporting an error from it.
    assert "SWITCHED:" in result.stdout, report


_STOLEN_HANDLER_SCRIPT = _XVFB_PRELUDE + r'''
import ctypes

import numpy

display = sys.argv[1]
server = start(display)
try:
    if not wait_for_socket(display) or wait_for_extension(display) is None:
        finish(2, "SETUP-FAILED: Xvfb never became reachable")

    # Another Xlib user in the process clears the process-global I/O
    # error handler -- which is what a toolkit does when it tears its
    # own connection down, and Tk runs in-process under recording/gui.
    # The extension is covered by Xlib's fatal default unless it
    # installs for every protected call rather than once per connect.
    libX11 = ctypes.CDLL("libX11.so.6")
    libX11.XSetIOErrorHandler.restype = ctypes.c_void_p
    libX11.XSetIOErrorHandler.argtypes = [ctypes.c_void_p]
    stolen_from = libX11.XSetIOErrorHandler(None)
    if not stolen_from:
        finish(2, "SETUP-FAILED: there was no handler installed to displace")

    stop(server)
    time.sleep(0.2)

    buf = numpy.zeros((16, 16, 4), "uint8")
    try:
        _linux_x11.screenshot(0, 0, buf)
    except RuntimeError as exc:
        dead = _linux_x11._display_cache_info()
        if dead["connected"]:
            finish(3, "KEPT-DEAD-CONNECTION: %r" % dead)
        # Surviving is not proof on its own -- reclaim the handler again
        # and check the extension had put the same function back, rather
        # than having got lucky some other way.
        reinstalled = libX11.XSetIOErrorHandler(None)
        if reinstalled != stolen_from:
            finish(3, "NOT-REINSTALLED: %r vs %r" % (reinstalled, stolen_from))
        finish(0, "RAISED: %s" % exc, "REINSTALLED", "SURVIVED")
    else:
        finish(3, "NO-ERROR: captured from a server that is gone")
finally:
    cleanup()
'''


def test_a_stolen_io_error_handler_is_reinstalled():
    """Xlib's I/O error handler is process-global and anyone can take it.

    Installing it once per connection meant every later cache hit ran
    under whatever handler was current -- so a disconnect never reached
    fg_io_error_handler() at all and the setjmp was inert. The
    reproduction is one XSetIOErrorHandler(NULL) between two calls.
    """
    result, report = _run_x_fault_script(_STOLEN_HANDLER_SCRIPT)
    assert result.returncode == 0, report
    assert "SURVIVED" in result.stdout, report
    assert "REINSTALLED" in result.stdout, report
    assert "cannot open X display" in result.stdout, report


# --------------------------------------------------------------------
# Two consequences of the connection being persistent rather than
# per-call: a delegation chain that can close into a loop, and an event
# queue that nothing reclaims.
# --------------------------------------------------------------------


def test_unread_events_do_not_pile_up_on_the_cached_connection():
    """MappingNotify reaches every client, and nothing here reads events.

    A per-call connection reclaimed its queue at XCloseDisplay. A cached
    one does not, so whatever the server pushes accumulates for the life
    of the process -- worst in exactly the long-running recording loop
    this cache exists to speed up.
    """
    xdisplay = pytest.importorskip("Xlib.display")

    _linux_x11.resolution()
    before = _linux_x11._display_cache_info()

    other = xdisplay.Display()
    try:
        mapping = other.get_pointer_mapping()
        for _ in range(25):
            # Every SetPointerMapping makes the server send a
            # MappingNotify to every client on the display, ours
            # included, whatever event mask it selected. Setting the
            # mapping it already has keeps the display untouched.
            other.set_pointer_mapping(mapping)
            other.sync()
            _linux_x11.resolution()
    finally:
        other.close()

    # One more call, so anything still sitting in the socket is pulled
    # into the queue and discarded rather than counted as a leak below.
    _linux_x11.resolution()
    after = _linux_x11._display_cache_info()

    # Count arrivals in a way that does not presuppose the fix: an event
    # that reached this connection was either discarded or is still
    # sitting in the queue. Asserting only on the discard counter would
    # make the guard fire first when the drain is removed, and the test
    # would never get to show the queue growing -- which is the symptom.
    arrived = (after["events_discarded"] - before["events_discarded"]
               + after["queued"])
    assert arrived > 0, (
        "no MappingNotify reached the cached connection, so this test "
        "cannot tell a drained queue from one nothing ever arrived on"
    )
    assert after["queued"] == 0, (
        "unread events are accumulating on the cached connection: "
        "{} still queued after {} arrived".format(after["queued"], arrived)
    )


_HANDLER_CYCLE_SCRIPT = _XVFB_PRELUDE + r'''
import ctypes

our_display, other_display = sys.argv[1], sys.argv[2]

libX11 = ctypes.CDLL("libX11.so.6")
HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
libX11.XSetIOErrorHandler.restype = HANDLER
libX11.XSetIOErrorHandler.argtypes = [HANDLER]
libX11.XOpenDisplay.restype = ctypes.c_void_p
libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
libX11.XSync.restype = ctypes.c_int
libX11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]


# The handler that was there first. Reaching it is the whole point: a
# delegation chain has to end somewhere.
@HANDLER
def base_handler(dpy):
    # finish() rather than a bare os._exit: an X I/O error handler must
    # not return, so this is the only chance to stop the servers this
    # script started. The successful path used to leak one here.
    finish(0, "BASE-REACHED")
    return 0


entries = [0]


# What a third party installs: it captures whatever handler is current
# and passes faults along to it. Harmless in itself -- unless the
# handler it captured is one that will hand the fault straight back.
@HANDLER
def delegating_handler(dpy):
    entries[0] += 1
    if entries[0] > 1:
        finish(9, "CYCLE: delegation came back round to this handler")
    displaced_by_us(dpy)
    finish(4, "UNREACHABLE: a delegate returned")
    return 0


ours = start(our_display)
other = start(other_display)
try:
    for display in (our_display, other_display):
        if not wait_for_socket(display):
            finish(2, "SETUP-FAILED: %s never appeared" % display)

    libX11.XSetIOErrorHandler(base_handler)

    if wait_for_extension(our_display) is None:
        finish(2, "SETUP-FAILED: %s never became reachable" % our_display)

    # Bounded retry rather than one shot: the socket existing does not
    # quite mean the server is accepting yet, and a lost race here would
    # fail the test against a correct extension.
    unrelated = None
    deadline = time.time() + 20.0
    while time.time() < deadline:
        unrelated = libX11.XOpenDisplay(other_display.encode())
        if unrelated:
            break
        time.sleep(0.05)
    if not unrelated:
        finish(2, "SETUP-FAILED: could not open %s" % other_display)

    # A third party installs its delegating handler between two
    # captures. Whatever it displaces becomes its predecessor.
    displaced_by_us = libX11.XSetIOErrorHandler(delegating_handler)

    # The second capture. If the extension leaves itself installed
    # globally, this is where it records the third party's wrapper as
    # its own predecessor and closes the loop.
    _linux_x11.resolution()

    stop(other)
    time.sleep(0.2)

    # A fatal I/O error on a connection that has nothing to do with the
    # cache. It must reach base_handler.
    libX11.XSync(unrelated, 0)
    finish(5, "NO-FAULT: the dead connection did not raise an I/O error")
finally:
    cleanup()
'''


def test_the_delegation_chain_cannot_close_into_a_loop():
    """Staying installed between calls let a chain become a cycle.

    A third party that installs a delegating handler in the gap between
    two captures captures *this* extension's handler as its predecessor.
    If the extension then re-installs and records that wrapper as its
    own predecessor, the two delegate to each other and an I/O error on
    an unrelated connection never reaches the handler that was there
    first. Testing "the handler I displaced is not literally me" only
    ever caught the direct case.

    The subprocess counts how many times the third-party handler is
    entered, so a cycle terminates the run with a marker instead of
    recursing without bound; the subprocess timeout is a second
    backstop.
    """
    result, report = _run_x_fault_script(_HANDLER_CYCLE_SCRIPT, displays=2)
    assert result.returncode == 0, report
    # Not just "it stopped" -- the fault has to have arrived at the
    # handler that was installed before any of this started.
    assert "BASE-REACHED" in result.stdout, report
    assert "CYCLE" not in result.stdout, report
