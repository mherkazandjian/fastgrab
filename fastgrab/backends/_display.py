"""Parsing and probing of ``$DISPLAY``.

Shared by the X11 backend's error reporting and by the test suite's
display gate. Deliberately stdlib-only and importable on every platform:
the gate consults it on the Windows and macOS runners, where
``fastgrab._linux_x11`` does not exist and must not be imported.

Nothing here speaks the X protocol. Establishing a connection is enough
to tell "a server is listening" from "nothing is there", which is the
only distinction either caller needs.

Nothing here raises, either. Both callers run it on a path where
something has *already* gone wrong, and a diagnostic that throws its own
exception over the original one is worse than no diagnostic at all.
"""
import os
import socket


# X listens on 6000 + display number when reached over TCP.
_X_TCP_BASE = 6000
_X_UNIX_PATH = "/tmp/.X11-unix/X{:d}"

# Transports that mean "this machine", whether they arrive as the
# protocol part of ``unix/:0`` or as the host part of ``unix:0``.
_LOCAL_NAMES = ("", "unix", "local")


def parse_display(spec):
    """Split a ``DISPLAY`` string into ``(host, display_number)``.

    The full Xlib form is ``[protocol/][host]:number[.screen]``, so this
    has to cope with more than a bare ``:0``:

    * an optional transport prefix — ``unix/:0``, ``tcp/box:0``. A
      ``unix``/``local`` transport is local no matter what the host says.
    * IPv6 literals, which Xlib writes bracketed (``[::1]:0``) and the
      socket API wants bare.
    * a trailing screen number, which does not change the socket.

    Returns ``None`` when *spec* is empty or does not parse. The host is
    ``""`` for a local connection.
    """
    if not spec:
        return None
    protocol = ""
    rest = spec
    if "/" in spec:
        protocol, _, rest = spec.partition("/")
    head, sep, tail = rest.rpartition(":")
    if not sep:
        return None
    number = tail.split(".", 1)[0]
    # isdecimal(), not isdigit(): the latter also accepts characters like
    # the superscript "²", which int() then refuses — turning a
    # malformed DISPLAY into a ValueError escaping the display gate
    # instead of a clean "unusable, skip".
    if not number.isdecimal():
        return None
    try:
        number = int(number)
    except ValueError:  # pragma: no cover - isdecimal() should preclude this
        return None
    host = head
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if protocol in ("unix", "local"):
        # An explicit local transport wins over whatever host was given.
        host = ""
    elif host in _LOCAL_NAMES:
        host = ""
    return host, number


def _probe_unix(number, timeout):
    if not hasattr(socket, "AF_UNIX"):
        return False
    path = _X_UNIX_PATH.format(number)
    # A NUL-prefixed name is the Linux abstract namespace, where modern X
    # servers may listen with no filesystem entry at all — so a missing
    # socket file is not on its own proof that nothing is listening.
    for candidate in (path, "\0" + path):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        except OSError:
            # Creating the socket can fail for the same reason
            # XOpenDisplay just did — a process out of file descriptors.
            # Report "cannot reach it" rather than raising EMFILE over
            # the caller's real error.
            return False
        try:
            sock.settimeout(timeout)
            sock.connect(candidate)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def _explicit_local_transport(spec):
    """Whether *spec* pins the transport to the local socket."""
    if not spec or "/" not in spec:
        return False
    return spec.partition("/")[0] in ("unix", "local")


def _probe_tcp(host, port, timeout):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, OverflowError, ValueError):
        # Covers refusal, timeout, unresolvable host (gaierror) and the
        # out-of-descriptors case, all of which mean "not reachable
        # here". OverflowError/ValueError catch a display number so large
        # that 6000 + N is not a port at all.
        return False


def probe_display(spec=None, timeout=1.0):
    """Return whether an X server for *spec* accepts a connection.

    *spec* defaults to ``$DISPLAY``. An unset, empty or unparseable value
    is reported as unreachable rather than optimistically available: the
    caller is deciding whether to run tests that need a display, and
    guessing "yes" turns a clean skip into a capture failure (issue #44).
    """
    if spec is None:
        spec = os.environ.get("DISPLAY")
    parsed = parse_display(spec)
    if parsed is None:
        return False
    host, number = parsed
    if host:
        return _probe_tcp(host, _X_TCP_BASE + number, timeout)
    if _probe_unix(number, timeout):
        return True
    # A bare ":N" names a display, not a transport: Xlib falls back to a
    # localhost TCP connection when the local socket is not there, so
    # stopping at the unix probe would call a working display
    # unreachable. An explicit "unix/" does pin the transport, and must
    # not fall back.
    if _explicit_local_transport(spec):
        return False
    return _probe_tcp("localhost", _X_TCP_BASE + number, timeout)


def describe_display(spec=None):
    """One-line description of the DISPLAY environment for an error message.

    Separates the three cases the C extension collapses into one string:
    not set at all, set to something unrecognisable, and set to a display
    that is (or is not) currently accepting connections.
    """
    if spec is None:
        spec = os.environ.get("DISPLAY")
    if spec is None:
        return "DISPLAY is not set"
    if not spec:
        return "DISPLAY is set but empty"
    if parse_display(spec) is None:
        return "DISPLAY={!r}, which is not a recognised display spec".format(spec)
    state = "reachable" if probe_display(spec) else "not reachable"
    return "DISPLAY={!r}, {}".format(spec, state)
