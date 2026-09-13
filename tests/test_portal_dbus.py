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

import numpy
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


# -------- PortalBackend against the same fixture --------

def test_the_backend_captures_through_the_portal(portal):
    """The whole path: handshake, node id, PipeWire frames, BGRA out."""
    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    try:
        width, height = backend.resolution()
        assert (width, height) == (320, 240)
        assert backend.bytes_per_pixel() == 4

        frame = numpy.zeros((height, width, 4), numpy.uint8)
        backend.screenshot(0, 0, frame)
        assert frame[..., 2].mean() > 200, "the fixture's red should be in byte 2"
        assert frame[..., 0].mean() < 50
    finally:
        backend.close()


def test_the_backend_crops_locally(portal):
    """The portal hands back a whole monitor; sub-regions are ours to cut."""
    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    try:
        frame = numpy.zeros((40, 60, 4), numpy.uint8)
        backend.screenshot(10, 20, frame)
        assert frame.shape == (40, 60, 4)
        assert frame[..., 2].mean() > 200
    finally:
        backend.close()


def test_a_region_outside_the_stream_is_refused(portal):
    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    try:
        with pytest.raises(ValueError, match="outside"):
            backend.screenshot(0, 0, numpy.zeros((1000, 1000, 4), numpy.uint8))
    finally:
        backend.close()


def test_construction_asks_for_nothing(portal):
    """Autodetection builds backends speculatively.

    A chooser dialog appearing because a program called Screenshot()
    would be indefensible, so the session opens on first use instead.
    """
    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    assert backend._session is None and backend._reader is None
    backend.close()


def test_one_backend_opens_one_session(portal):
    """Consent belongs to the session, so reuse must not re-ask."""
    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    try:
        backend.resolution()
        session = backend._session
        for _ in range(3):
            backend.screenshot(0, 0, numpy.zeros((240, 320, 4), numpy.uint8))
        assert backend._session is session, "a capture re-opened the session"
    finally:
        backend.close()


def test_the_reader_is_given_the_portals_own_pipewire_socket(portal):
    """The node id is only meaningful on the portal's connection.

    Nothing in this fixture can tell the two connections apart -- the
    fake hands back a socket to the very daemon the ambient environment
    would have reached anyway -- so a behavioural test here would pass
    just as well with the descriptor thrown away. It is asserted
    structurally instead, because on a real desktop the difference is
    the whole point: the ambient socket shows a different graph, and
    the same node id on it is a different node, or nothing.
    """
    import socket

    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    seen = {}
    real = backend._reader_class

    def spy(node_id, *args, **kwargs):
        seen["fd"] = kwargs.get("fd")
        return real(node_id, *args, **kwargs)

    backend._reader_class = spy
    try:
        backend.resolution()
        assert seen.get("fd") is not None, (
            "the reader was built without the portal's descriptor and "
            "fell back to the ambient PipeWire socket"
        )
        # A real descriptor, not the index that came off the wire:
        # D-Bus type 'h' is an index into the message's FD list, and
        # using it as an fd reads whatever unrelated file holds that
        # number -- which on a small process is usually a socket too,
        # hence checking the family rather than just openability.
        duplicate = os.dup(seen["fd"])
        try:
            sock = socket.socket(fileno=duplicate)
        except OSError:
            os.close(duplicate)
            pytest.fail("the descriptor handed to the reader is not a socket")
        with sock:
            assert sock.family == socket.AF_UNIX
    finally:
        backend.close()


def test_a_discarded_backend_releases_its_pipeline(portal):
    """The two-line form never calls close().

    ``Screenshot(backend='portal').capture()`` drops the backend on the
    same line, and the library documents that shape. Without a
    finalizer it leaks a PLAYING GStreamer pipeline and the portal's
    descriptor for the life of the process.
    """
    import gc

    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    closed = []
    real = backend._reader_class

    def spy(node_id, *args, **kwargs):
        reader = real(node_id, *args, **kwargs)
        original = reader.close

        def close():
            closed.append(True)
            return original()

        reader.close = close
        return reader

    backend._reader_class = spy
    backend.resolution()
    # The finalizer holds this dict, so it outlives the backend and
    # shows what the teardown actually did.
    state = backend.__dict__
    del backend
    gc.collect()

    assert closed, "a discarded backend left its GStreamer pipeline running"
    assert state["_fd"] is None, "a discarded backend leaked the portal's fd"
    assert state["_session"] is None, "a discarded backend leaked the session"


def test_refresh_renegotiates_the_stream_without_re_asking(portal):
    """Consent belongs to the session, so refresh() must not touch it.

    It drops the stream and the portal descriptor -- single-use per
    consumer connection, so the next capture has to ask for a fresh one
    -- and leaves the session, and therefore the approval, alone.
    """
    from fastgrab.backends.portal import PortalBackend
    portal()
    backend = PortalBackend(timeout=30)
    try:
        backend.resolution()
        session = backend._session
        assert backend._fd is not None

        backend.refresh()
        assert backend._session is session, "refresh() re-opened the session"
        assert backend._reader is None, "refresh() kept the old stream"
        assert backend._fd is None, "refresh() kept the spent descriptor"

        asked = []
        spent = session.open_pipewire_remote

        def spy():
            asked.append(True)
            return spent()

        session.open_pipewire_remote = spy
        try:
            backend.screenshot(0, 0, numpy.zeros((240, 320, 4), numpy.uint8))
        except RuntimeError as exc:
            # Expected against this fixture, and not a backend bug. Its
            # source is `pipewiresink mode=provide`, which exits the
            # moment its consumer disconnects -- so closing the reader
            # above took the node off the graph with it. A real
            # compositor keeps its node across a client reconnect. What
            # is under test here is our half: that renegotiation was
            # attempted, and attempted with a new descriptor rather than
            # the spent one.
            assert "target not found" in str(exc), (
                "refresh() failed for some reason other than the "
                "fixture's provider having exited: %s" % exc
            )
        assert asked, "the renegotiated stream reused the spent descriptor"
    finally:
        backend.close()


def test_construction_probes_for_the_optional_dependencies(monkeypatch):
    """_autodetect() decides which backend to use by constructing one.

    So a constructor that succeeds without PyGObject and GStreamer
    claims every GNOME and KDE session for a backend that cannot work,
    and the XWayland fallback that would have worked never runs -- the
    failure surfaces from capture() instead, outside the handler that
    exists to catch exactly this.
    """
    from fastgrab.backends import _pipewire
    from fastgrab.backends.portal import PortalBackend

    def missing(*_args, **_kwargs):
        raise RuntimeError("GStreamer has no 'pipewiresrc' element")

    monkeypatch.setattr(_pipewire, "_require_gst", missing)
    with pytest.raises(RuntimeError, match="pipewiresrc"):
        PortalBackend()


@pytest.mark.parametrize("value,expected", [
    ("none", 0), ("transient", 1), ("persistent", 2), (None, 1),
])
def test_the_persist_mode_can_be_chosen(value, expected, monkeypatch):
    from fastgrab.backends.portal import PortalBackend, _persist_mode
    monkeypatch.delenv("FASTGRAB_PORTAL_PERSIST", raising=False)
    assert _persist_mode(value) == expected


def test_the_persist_mode_can_come_from_the_environment(monkeypatch):
    from fastgrab.backends.portal import _persist_mode
    monkeypatch.setenv("FASTGRAB_PORTAL_PERSIST", "persistent")
    assert _persist_mode() == 2
    monkeypatch.setenv("FASTGRAB_PORTAL_PERSIST", "nonsense")
    with pytest.raises(ValueError, match="FASTGRAB_PORTAL_PERSIST"):
        _persist_mode()
