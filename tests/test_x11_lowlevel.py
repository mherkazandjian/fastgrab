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


def _free_display_number():
    for number in range(90, 110):
        if os.path.exists("/tmp/.X11-unix/X%d" % number):
            continue
        if os.path.exists("/tmp/.X%d-lock" % number):
            continue
        return number
    return None


# Run out-of-process on purpose. The failure this guards against is the
# interpreter *exiting*, which an in-process test cannot report — it
# would take the whole pytest session down with it. Out here a
# regression shows up as a non-zero exit status plus Xlib's "XIO: fatal
# IO error" on stderr, both of which the assertion prints.
_DEAD_SERVER_SCRIPT = r'''
import os
import subprocess
import sys
import time

import numpy

from fastgrab import _linux_x11

display = sys.argv[1]
server = subprocess.Popen(
    ["Xvfb", display, "-screen", "0", "320x240x24", "-ac"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
os.environ["DISPLAY"] = display
try:
    deadline = time.time() + 15.0
    while True:
        try:
            _linux_x11.resolution()
            break
        except RuntimeError:
            if time.time() > deadline:
                print("SETUP-FAILED: Xvfb never became reachable")
                raise SystemExit(2)
            time.sleep(0.05)

    buf = numpy.zeros((16, 16, 4), "uint8")
    _linux_x11.screenshot(0, 0, buf)
    live = _linux_x11._display_cache_info()
    if not (live["connected"] and live["display"] == display):
        print("SETUP-FAILED: not connected to the private server:", live)
        raise SystemExit(2)

    server.terminate()
    server.wait(timeout=15)
    time.sleep(0.2)

    # The cached connection is now a dead socket. Xlib's default I/O
    # error handler would print and call exit() right here.
    try:
        _linux_x11.screenshot(0, 0, buf)
    except RuntimeError as exc:
        dead = _linux_x11._display_cache_info()
        if dead["connected"]:
            print("KEPT-DEAD-CONNECTION:", dead)
            raise SystemExit(3)
        print("RAISED:", exc)
        print("SURVIVED")
    else:
        print("NO-ERROR: captured from a server that is gone")
        raise SystemExit(3)
finally:
    if server.poll() is None:
        server.kill()
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
    if not shutil.which("Xvfb"):
        pytest.skip("Xvfb is not installed; cannot run a disposable X server")
    number = _free_display_number()
    if number is None:
        pytest.skip("no free X display number for a disposable server")

    result = subprocess.run(
        [sys.executable, "-c", _DEAD_SERVER_SCRIPT, ":%d" % number],
        capture_output=True, text=True, timeout=120,
    )
    report = "exit={}\nstdout:\n{}\nstderr:\n{}".format(
        result.returncode, result.stdout, result.stderr,
    )
    assert result.returncode == 0, report
    assert "SURVIVED" in result.stdout, report
    # Not merely "did not crash": the connection has to have been given
    # up, so the next call reconnects rather than reusing a dead socket.
    assert "cannot open X display" in result.stdout, report
