"""Parsing and probing of ``$DISPLAY``.

Shared by the X11 backend's error reporting and by the test suite's
display gate. Deliberately stdlib-only and importable on every platform:
the gate consults it on the Windows and macOS runners, where
``fastgrab._linux_x11`` does not exist and must not be imported.

Nothing here speaks the X protocol. Establishing a connection is enough
to tell "a server is listening" from "nothing is there", which is the
only distinction either caller needs.
"""
import os
import socket


# X listens on 6000 + display number when reached over TCP.
_X_TCP_BASE = 6000
_X_UNIX_PATH = "/tmp/.X11-unix/X{:d}"


def parse_display(spec):
    """Split a ``DISPLAY`` string into ``(host, display_number)``.

    ``None`` when *spec* is empty or is not of the ``[host]:number[.screen]``
    form. The host is ``""`` for a local connection, which covers both the
    bare ``:0`` and the explicit ``unix:0`` spellings.
    """
    if not spec:
        return None
    head, sep, tail = spec.rpartition(":")
    if not sep:
        return None
    # ``:0.1`` selects screen 1 of display 0; the screen does not affect
    # which socket the connection goes to.
    number = tail.split(".", 1)[0]
    if not number.isdigit():
        return None
    host = "" if head in ("", "unix") else head
    return host, int(number)


def _probe_unix(number, timeout):
    path = _X_UNIX_PATH.format(number)
    if not hasattr(socket, "AF_UNIX"):
        return False
    # A NUL-prefixed name is the Linux abstract namespace, where modern X
    # servers may listen with no filesystem entry at all — so a missing
    # socket file is not on its own proof that nothing is listening.
    for candidate in (path, "\0" + path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(candidate)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def _probe_tcp(host, port, timeout):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
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
    return _probe_unix(number, timeout)


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
