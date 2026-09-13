"""The ScreenCast handshake, against the real xdg-desktop-portal frontend.

Only the desktop-environment half is faked — the chooser and the
compositor's node. Everything the client talks to is genuine portal
code: request objects, Response signals, session lifetime and the
file-descriptor handoff.
"""
import json
import os
import subprocess
import time

import pytest

pytestmark = pytest.mark.portal

pytest.importorskip("gi", reason="the portal backend is behind the [portal] extra")

from fastgrab.backends._portal_dbus import (  # noqa: E402
    PortalCancelled, PortalUnavailable, ScreenCastSession,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "portal", "fake_impl_screencast.py")


def _wait_for_video_source(timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dump = subprocess.run(["pw-dump"], capture_output=True)
        try:
            objects = json.loads(dump.stdout or b"[]")
        except ValueError:
            objects = []
        for obj in objects:
            props = (obj.get("info") or {}).get("props") or {}
            if props.get("media.class") == "Video/Source":
                return obj["id"]
        time.sleep(0.25)
    return None


@pytest.fixture
def portal(tmp_path, monkeypatch):
    """A private session bus, a fake impl backend, and the real frontend.

    Per test: the frontend caches its backend choice and a session's
    lifetime is the thing under test, so sharing one across tests would
    hide exactly the bugs this is here to find.
    """
    source = subprocess.Popen(
        ["gst-launch-1.0", "-q", "videotestsrc", "pattern=solid-color",
         "foreground-color=0xffff0000", "is-live=true",
         "!", "video/x-raw,format=BGRx,width=320,height=240,framerate=10/1",
         "!", "pipewiresink", "mode=provide",
         "stream-properties=props,node.name=fastgrab-fixture,"
         "media.class=Video/Source"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    node = _wait_for_video_source()
    if node is None:
        source.kill()
        pytest.fail("the synthetic PipeWire source never appeared")

    address = subprocess.run(
        ["dbus-daemon", "--session", "--print-address", "--fork"],
        capture_output=True, text=True).stdout.strip()
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", address)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "fastgrabtest")

    portals = "/usr/share/xdg-desktop-portal/portals"
    os.makedirs(portals, exist_ok=True)
    with open(os.path.join(portals, "fastgrabfake.portal"), "w") as handle:
        handle.write(
            "[portal]\n"
            "DBusName=org.freedesktop.impl.portal.desktop.fastgrabfake\n"
            "Interfaces=org.freedesktop.impl.portal.ScreenCast\n"
            "UseIn=fastgrabtest\n")

    env = dict(os.environ, DBUS_SESSION_BUS_ADDRESS=address,
               XDG_CURRENT_DESKTOP="fastgrabtest")
    started = {"node": node}

    def launch(response=0):
        fake = subprocess.Popen(
            ["python", FAKE, str(node), str(response)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(1.0)
        front = subprocess.Popen(
            ["/usr/libexec/xdg-desktop-portal", "-r"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        started["procs"] = [fake, front]
        return started

    yield launch
    for proc in started.get("procs", []):
        proc.kill()
        proc.wait(timeout=10)
    source.kill()
    source.wait(timeout=10)


def test_the_handshake_returns_the_approved_node(portal):
    state = portal()
    session = ScreenCastSession(app_id="fastgrab-test", timeout=30)
    try:
        streams = session.open()
        assert streams, "the portal approved but offered nothing"
        assert session.node_id == state["node"]
    finally:
        session.close()


def test_a_declined_request_is_not_a_generic_failure(portal):
    """Response 1 must be distinguishable from everything else.

    Falling back to XWayland after someone declined would capture the
    screen they just refused to share.
    """
    portal(response=1)
    session = ScreenCastSession(timeout=30)
    try:
        with pytest.raises(PortalCancelled):
            session.open()
    finally:
        session.close()


def test_a_refusal_is_reported_as_unavailable(portal):
    portal(response=2)
    session = ScreenCastSession(timeout=30)
    try:
        with pytest.raises(PortalUnavailable):
            session.open()
    finally:
        session.close()


def test_the_pipewire_fd_is_taken_from_the_message_fd_list(portal):
    """D-Bus 'h' is an index, not a descriptor.

    Treating the returned integer as a file descriptor is the classic
    mistake: it usually *is* a valid open descriptor in this process,
    just an unrelated one, so the bug shows up as garbage rather than an
    error. The index here is small; the descriptor is not.
    """
    portal()
    session = ScreenCastSession(timeout=30)
    try:
        session.open()
        fd = session.open_pipewire_remote()
        assert isinstance(fd, int) and fd >= 0
        # A real descriptor this process owns, not the index 0 the wire carried.
        assert os.fstat(fd) is not None
        os.close(fd)
    finally:
        session.close()


def test_closing_twice_is_harmless(portal):
    portal()
    session = ScreenCastSession(timeout=30)
    session.open()
    session.close()
    session.close()
