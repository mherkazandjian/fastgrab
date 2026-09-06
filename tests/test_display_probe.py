"""Tests for the DISPLAY parser/prober behind the X11 error messages.

Pure stdlib sockets and string handling, so these run on every platform —
which matters, because the test suite's own display gate imports this
module on the Windows and macOS runners too.
"""
import socket

import pytest

from fastgrab.backends._display import (
    describe_display,
    parse_display,
    probe_display,
)

# Nothing here needs a display server; several cases assert the opposite.
pytestmark = pytest.mark.no_display


@pytest.mark.parametrize("spec, expected", [
    (":0", ("", 0)),
    (":0.0", ("", 0)),
    (":77", ("", 77)),
    (":0.1", ("", 0)),          # screen suffix does not change the socket
    ("unix:0", ("", 0)),        # explicit local spelling
    ("localhost:10", ("localhost", 10)),
    ("localhost:10.0", ("localhost", 10)),
    ("192.168.1.5:0", ("192.168.1.5", 0)),
])
def test_parse_display_accepts_the_documented_forms(spec, expected):
    assert parse_display(spec) == expected


@pytest.mark.parametrize("spec", [
    None, "", "not-a-display", ":", ":abc", "localhost", "localhost:",
])
def test_parse_display_rejects_what_it_cannot_read(spec):
    assert parse_display(spec) is None


@pytest.mark.parametrize("spec", ["", "garbage", ":abc"])
def test_probe_reports_unusable_specs_as_unreachable(spec):
    # Optimism here is what turns a skip into a capture failure: the
    # caller uses this to decide whether to run display tests at all.
    assert probe_display(spec) is False


def test_probe_finds_a_listening_server():
    # Stand in for an X server: anything accepting on 6000 + N is enough,
    # since the probe deliberately speaks no X protocol.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        if port < 6000:  # pragma: no cover - ephemeral ports are far above
            pytest.skip("ephemeral port below the X TCP base")
        assert probe_display("127.0.0.1:{:d}".format(port - 6000)) is True
    finally:
        listener.close()


def test_probe_reports_a_closed_port_as_unreachable():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()  # nothing is listening on it now
    if port < 6000:  # pragma: no cover
        pytest.skip("ephemeral port below the X TCP base")
    assert probe_display(
        "127.0.0.1:{:d}".format(port - 6000), timeout=0.5
    ) is False


def test_probe_reports_an_absent_local_server_as_unreachable():
    # :77 is the number the C extension's own regression test uses for
    # "definitely not there".
    assert probe_display(":77", timeout=0.5) is False


def test_describe_separates_unset_from_empty(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    assert describe_display() == "DISPLAY is not set"
    monkeypatch.setenv("DISPLAY", "")
    assert describe_display() == "DISPLAY is set but empty"


def test_describe_flags_an_unrecognised_spec():
    assert "not a recognised display spec" in describe_display("nonsense")


def test_describe_reports_reachability():
    assert "not reachable" in describe_display(":77")


def test_probe_falls_back_to_the_environment(monkeypatch):
    # A None spec means "whatever DISPLAY says", which is how both the
    # error message and the test gate call it.
    monkeypatch.delenv("DISPLAY", raising=False)
    assert probe_display() is False
    monkeypatch.setenv("DISPLAY", ":77")
    assert probe_display(timeout=0.5) is False
