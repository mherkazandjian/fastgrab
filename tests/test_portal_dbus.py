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
    REQUEST_IFACE, PortalCancelled, PortalUnavailable, ScreenCastSession,
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

    def launch(response=0, extra_env=None):
        fake = subprocess.Popen(
            ["python", FAKE, str(node), str(response)],
            env=dict(env, **(extra_env or {})),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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


def test_the_handshake_works_from_a_thread_with_its_own_glib_context(portal):
    """Any GTK or Gio program can be driving this from a worker thread.

    signal_subscribe() delivers on the thread-default GLib context as
    it stands when the subscription is made; a plain MainLoop() and
    timeout_add() use the global default one. On the main thread those
    are the same object and the difference never shows. Push a context
    of your own first -- which is ordinary practice in threaded GLib
    code -- and the portal's answer arrives somewhere the loop never
    runs, so every request times out despite the portal having replied
    at once.
    """
    import threading

    from gi.repository import GLib

    portal()
    outcome = {}

    def run():
        context = GLib.MainContext.new()
        context.push_thread_default()
        try:
            session = ScreenCastSession(app_id="fastgrab-test", timeout=15)
            try:
                outcome["streams"] = session.open()
            finally:
                session.close()
        except BaseException as exc:      # noqa: BLE001 - reported below
            outcome["error"] = exc
        finally:
            context.pop_thread_default()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(90)

    assert not thread.is_alive(), "the handshake never returned"
    assert "error" not in outcome, "handshake failed: %r" % (outcome["error"],)
    assert outcome["streams"], "no streams came back"


def test_an_abandoned_request_is_closed():
    """Unsubscribing stops us listening; it does not stop the request.

    The portal keeps a Request object alive until it answers or the
    client closes it. Walking away from a timeout leaves it live at the
    other end, still able to create a session nobody holds -- and any
    other Gio user, or a retained transient token, keeps the connection
    and therefore the orphan alive.

    Driven against a stand-in connection rather than the real portal.
    Spying on the shared Gio bus poisons every portal test that follows:
    the connection is process-wide, and holding it across the fixture's
    teardown leaves later tests talking to a bus whose portal is gone.
    """
    class FakeConnection(object):
        def __init__(self):
            self.calls = []

        def get_unique_name(self):
            return ":1.99"

        def signal_subscribe(self, *_args, **_kwargs):
            return 1

        def signal_unsubscribe(self, _subscription):
            pass

        def call_sync(self, _bus, _path, iface, method, *_a, **_k):
            self.calls.append((iface, method))
            return None

    session = ScreenCastSession(app_id="fastgrab-test", timeout=0.2)
    connection = FakeConnection()
    session._conn = connection

    with pytest.raises(PortalUnavailable, match="never answered"):
        session.open()

    assert (REQUEST_IFACE, "Close") in connection.calls, (
        "the abandoned request was never closed; it stays live at the "
        "portal and can still create a session. Calls seen: %s"
        % (connection.calls,)
    )


# -------- remembering consent --------

@pytest.fixture
def clean_token_store(tmp_path, monkeypatch):
    """An isolated token file, and no leftovers in the process cache.

    The in-process cache is module state, so without this a token from
    one test would satisfy the next and the assertions would pass for
    the wrong reason.
    """
    from fastgrab.backends import portal as portal_module

    portal_module._PROCESS_TOKEN.clear()
    path = tmp_path / "portal-restore-token"
    monkeypatch.setenv("FASTGRAB_PORTAL_TOKEN_FILE", str(path))
    yield path
    portal_module._PROCESS_TOKEN.clear()


def _selections(state_file):
    """What the desktop backend was handed, one dict per SelectSources."""
    out = []
    for line in state_file.read_text().strip().splitlines():
        fields = line.split()
        out.append({"token": fields[1], "persist_mode": fields[2],
                    "restored": fields[3] == "restored"})
    return out


def _offered(state_file):
    """What it was handed on the *first* SelectSources."""
    first = _selections(state_file)[0]
    return {"token": first["token"], "persist_mode": first["persist_mode"]}


def test_a_persistent_approval_is_kept_for_the_next_process(
    portal, tmp_path, clean_token_store
):
    """Otherwise 'remember this' silently means 'ask me again'.

    persist_mode was already being sent, but the token the portal hands
    back in return was dropped on the floor -- and the token is the
    whole mechanism. Nothing carried over, whichever mode was chosen.
    """
    from fastgrab.backends.portal import PortalBackend

    state = tmp_path / "fake-state"
    portal(extra_env={"FASTGRAB_FAKE_STATE": str(state),
                      "FASTGRAB_FAKE_TOKEN": "tok-1"})
    backend = PortalBackend(persist="persistent", timeout=30)
    try:
        backend.resolution()
    finally:
        backend.close()

    assert _offered(state) == {"token": "-", "persist_mode": "2"}, (
        "the first session should offer no token and ask to persist"
    )
    assert clean_token_store.read_text() == "tok-1"
    mode = os.stat(str(clean_token_store)).st_mode & 0o777
    assert mode == 0o600, (
        "the token reopens the screen share without asking, so it must "
        "not be readable by anyone else (found %s)" % oct(mode)
    )


def test_a_persisted_approval_actually_restores_without_asking(
    portal, tmp_path, clean_token_store
):
    """The other persistence tests prove plumbing, not restoration.

    They check that a token is offered, rotated and stored. None of
    them shows a restore ever *succeeding*, because the fake used to
    return an arbitrary `restore_token` string -- and that is not the
    impl contract. A desktop backend returns `restore_data (suv)`; the
    frontend stores it, issues the client a UUID of its own, and on the
    next session resolves that UUID back into restore_data for the
    implementation. A made-up string creates no restorable permission
    at all and fails the frontend's UUID validation when offered back,
    so the persistence path was passing without ever restoring.

    This drives two sessions and asserts the second one arrives at the
    desktop already restored -- which is what "do not ask me again"
    means.
    """
    from fastgrab.backends import portal as portal_module
    from fastgrab.backends.portal import PortalBackend

    state = tmp_path / "fake-state"
    portal(extra_env={"FASTGRAB_FAKE_STATE": str(state),
                      "FASTGRAB_FAKE_RESTORE": "1"})

    first = PortalBackend(persist="persistent", timeout=30)
    try:
        first.resolution()
    finally:
        first.close()

    assert clean_token_store.exists(), (
        "no token was stored: the frontend issues one only when the "
        "desktop backend returns restore_data, so nothing here is "
        "restorable"
    )
    assert clean_token_store.read_text().strip(), "the stored token is empty"
    # What a *later process* starts from: nothing cached, token on disk.
    portal_module._PROCESS_TOKEN.clear()

    second = PortalBackend(persist="persistent", timeout=30)
    try:
        try:
            second.resolution()
        except RuntimeError:
            # Expected here and not a backend failure: the fixture's
            # provider is `pipewiresink mode=provide` and exits with its
            # first consumer, so there is no node left to read. The
            # handshake is what this test is about, and it is recorded
            # below either way.
            pass
    finally:
        second.close()

    seen = _selections(state)
    assert len(seen) >= 2, "the second session never reached the desktop"
    assert not seen[0]["restored"], "the first session restored something"
    assert seen[1]["restored"], (
        "the second session reached the desktop with no restore data, so "
        "the user would have been asked again despite persist='persistent'"
    )


def test_a_stale_token_is_offered_then_dropped_rather_than_wedging_capture(
    portal, tmp_path, clean_token_store
):
    """A token the desktop no longer honours must not be fatal.

    Asserted at the session boundary, because the token never reaches
    the desktop backend: xdg-desktop-portal owns the token database, so
    it resolves the client's token itself and hands the implementation
    its own restore data. An entry the frontend does not recognise --
    exactly what a token from a previous boot looks like here -- is
    rejected outright.

    Without the retry that is a permanent failure: every capture from
    then on dies on a stored file the user has no reason to know about.
    So the sequence is offer it, and on refusal forget it and ask
    properly, once. The portal also rotates the token on success, so
    the one that comes back has to replace the one that went in.
    """
    from fastgrab.backends.portal import PortalBackend

    clean_token_store.write_text("tok-from-last-boot")
    state = tmp_path / "fake-state"
    portal(extra_env={"FASTGRAB_FAKE_STATE": str(state),
                      "FASTGRAB_FAKE_TOKEN": "tok-2"})
    backend = PortalBackend(persist="persistent", timeout=30)
    calls = []
    real = backend._session_class

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    backend._session_class = spy
    try:
        assert backend.resolution() == (320, 240), (
            "a stale token left capture broken"
        )
    finally:
        backend.close()

    assert calls[0].get("restore_token") == "tok-from-last-boot", (
        "the stored token was never offered, so persist='persistent' "
        "would prompt every time regardless"
    )
    assert len(calls) == 2, (
        "expected exactly one retry after the refusal, got %d session(s)"
        % len(calls)
    )
    assert calls[1].get("restore_token") is None, (
        "the retry offered the same rejected token again"
    )
    assert clean_token_store.read_text() == "tok-2", (
        "the rotated token was not kept, so the next run would offer a "
        "spent one"
    )


def test_writing_over_a_permissive_token_file_makes_it_private(
    tmp_path, monkeypatch
):
    """O_CREAT's mode only applies when the file is created.

    A token file already sitting there group- or world-readable --
    copied between machines, restored from a backup, written by an
    older version -- would otherwise keep those permissions and be
    handed the rotated capability anyway. It reopens the screen share
    without asking, so nobody else may read it.

    Driven straight at _write_token rather than through a capture: the
    stale-token recovery path *deletes* the file before rewriting it,
    so the end-to-end route never reaches the overwrite this is about
    and passes whether or not the bug is present.
    """
    from fastgrab.backends import portal as portal_module

    path = tmp_path / "portal-restore-token"
    path.write_text("older-token")
    os.chmod(str(path), 0o644)
    monkeypatch.setenv("FASTGRAB_PORTAL_TOKEN_FILE", str(path))

    portal_module._write_token("tok-rotated")

    assert path.read_text() == "tok-rotated"
    mode = os.stat(str(path)).st_mode & 0o777
    assert mode == 0o600, (
        "an already-permissive token file kept mode %s and was given "
        "the rotated capability" % oct(mode)
    )


def test_a_transient_token_keeps_its_bus_connection_alive(
    portal, tmp_path, clean_token_store
):
    """A transient permission belongs to the D-Bus client.

    Gio's shared session bus does not outlive its last reference:
    measured, dropping it and asking again yields a connection with a
    new unique name. In a script whose only Gio user is fastgrab,
    releasing the last Screenshot would leave a remembered token the
    desktop no longer recognises, so the next one prompts after all --
    the opposite of what transient consent is for.
    """
    import gc

    from gi.repository import Gio

    from fastgrab.backends import portal as portal_module
    from fastgrab.backends.portal import PortalBackend

    portal(extra_env={"FASTGRAB_FAKE_TOKEN": "tok-transient"})
    backend = PortalBackend(persist="transient", timeout=30)
    backend.resolution()
    during = backend._session.connection.get_unique_name()
    backend.close()
    del backend
    gc.collect()

    assert portal_module._PROCESS_TOKEN.get("token") == "tok-transient"
    assert portal_module._PROCESS_TOKEN.get("connection") is not None, (
        "the token was remembered without the connection it belongs to"
    )
    after = Gio.bus_get_sync(Gio.BusType.SESSION, None).get_unique_name()
    assert after == during, (
        "the bus client changed from %s to %s, so the remembered "
        "transient token is no longer valid" % (during, after)
    )


def test_token_recovery_is_bounded_even_when_the_file_cannot_be_deleted(
    portal, tmp_path, clean_token_store, monkeypatch
):
    """Forgetting a rejected token is not the same as deleting it.

    _forget_token() swallows a failed unlink -- an unwritable state
    directory, a read-only home -- so a retry that simply re-reads the
    token file gets the same rejected token back and opens a portal
    session per attempt until RecursionError. The retry has to stop
    loading tokens explicitly, not trust the deletion.
    """
    from fastgrab.backends.portal import PortalBackend

    clean_token_store.write_text("tok-that-will-be-rejected")
    state = tmp_path / "fake-state"
    portal(extra_env={"FASTGRAB_FAKE_STATE": str(state),
                      "FASTGRAB_FAKE_TOKEN": "tok-fresh"})

    def refuse(_path):
        raise OSError("read-only file system")

    monkeypatch.setattr(os, "remove", refuse)

    backend = PortalBackend(persist="persistent", timeout=30)
    calls = []
    real = backend._session_class

    def spy(**kwargs):
        calls.append(kwargs)
        # Fails fast and says why, rather than letting an unbounded
        # recovery loop run a real portal handshake a thousand times.
        assert len(calls) <= 3, "token recovery is looping"
        return real(**kwargs)

    backend._session_class = spy
    try:
        assert backend.resolution() == (320, 240)
    finally:
        backend.close()

    assert len(calls) == 2, (
        "expected the stale token then exactly one retry without it, "
        "got %d sessions" % len(calls)
    )
    assert calls[1].get("restore_token") is None


def test_asking_every_time_stores_nothing(portal, tmp_path, clean_token_store):
    """persist='none' is a privacy choice and has to be honoured."""
    from fastgrab.backends.portal import PortalBackend

    state = tmp_path / "fake-state"
    portal(extra_env={"FASTGRAB_FAKE_STATE": str(state),
                      "FASTGRAB_FAKE_TOKEN": "tok-3"})
    backend = PortalBackend(persist="none", timeout=30)
    try:
        backend.resolution()
    finally:
        backend.close()

    assert _offered(state) == {"token": "-", "persist_mode": "0"}
    assert not clean_token_store.exists(), (
        "a token was written to disk even though the user asked to be "
        "prompted every time"
    )


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


def test_construction_refuses_when_no_screencast_portal_is_present(portal):
    """Having the libraries says nothing about the desktop.

    The fixture is requested but deliberately **not** launched, so the
    session bus here is real and carries no ScreenCast implementation --
    a Wayland session that installed the extra and has no portal. Its
    XWayland works fine, and auto-detection must leave it alone rather
    than claim it and fail later from capture().

    The frontend is still D-Bus-activatable and does start; what it does
    not do is export the interface, which is what the probe reads.
    """
    from fastgrab.backends._portal_dbus import PortalUnavailable
    from fastgrab.backends.portal import PortalBackend

    with pytest.raises(PortalUnavailable, match="ScreenCast"):
        PortalBackend(timeout=5)


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
