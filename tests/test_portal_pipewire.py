"""Frame reading from a PipeWire node, with no compositor and no GPU.

Run by the ``test-portal`` compose service, which brings up a private
PipeWire graph carrying one synthetic Video/Source. That fixture exists
because the portal backend's other half — the xdg-desktop-portal
handshake that yields a node id — is blocked upstream (issue #54), and
this half need not be.
"""
import json
import os
import subprocess
import time

import numpy
import pytest

pytestmark = pytest.mark.portal

# Skipped where the [portal] extra is absent — the default test image has
# no PyGObject and would fail collection for the whole suite. Inside the
# test-portal service the import succeeds, and a *missing fixture* there
# still fails loudly rather than skipping; see the node_id fixture.
pytest.importorskip(
    "gi", reason="the portal backend is behind the [portal] extra"
)

from fastgrab.backends._pipewire import PipeWireVideoReader  # noqa: E402


def _fixture_node_id(timeout=20.0):
    """The id of the synthetic Video/Source, waiting for it to appear.

    Bounded polling rather than one look: the fixture's provider exits
    when its last consumer disconnects and is restarted by a supervisor,
    so between tests there is a window with no node on the graph. A
    single check lands in that window and reports the fixture as absent.
    """
    deadline = time.monotonic() + timeout
    while True:
        dump = subprocess.run(["pw-dump"], capture_output=True)
        try:
            objects = json.loads(dump.stdout or b"[]")
        except ValueError:
            objects = []
        for obj in objects:
            props = (obj.get("info") or {}).get("props") or {}
            if props.get("media.class") == "Video/Source":
                return obj["id"]
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


@pytest.fixture
def node_id():
    """A synthetic Video/Source, started and stopped for this one test.

    Not shared. `pipewiresink mode=provide` errors with "all buffers have
    been removed / PipeWire link to remote node was destroyed" and exits
    as soon as its consumer disconnects, so one shared provider survives
    exactly one test and every test after it fails with "target not
    found" — which reads like a bug in the reader.
    """
    width = int(os.environ.get("FASTGRAB_FIXTURE_WIDTH", 320))
    height = int(os.environ.get("FASTGRAB_FIXTURE_HEIGHT", 240))
    proc = subprocess.Popen(
        ["gst-launch-1.0", "-q",
         "videotestsrc", "pattern=solid-color",
         "foreground-color=0xffff0000", "is-live=true",
         "!", "video/x-raw,format=BGRx,width=%d,height=%d,framerate=10/1"
         % (width, height),
         "!", "pipewiresink", "mode=provide",
         "stream-properties=props,node.name=fastgrab-fixture,"
         "media.class=Video/Source"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        found = _fixture_node_id()
        if found is None:
            proc.kill()
            pytest.fail(
                "no Video/Source appeared on the PipeWire graph. This "
                "fails rather than skips: a silently absent dependency "
                "is how a backend stops being tested without anyone "
                "noticing. stderr: %s"
                % (proc.stderr.read() or b"")[-300:].decode("utf-8", "replace")
            )
        yield found
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_a_frame_comes_back_as_bgra(node_id):
    with PipeWireVideoReader(node_id) as reader:
        frame = reader.read()
    assert frame.dtype == numpy.uint8
    assert frame.ndim == 3 and frame.shape[2] == 4


def test_the_size_comes_from_the_negotiated_caps(node_id):
    """Not from any declared geometry — the stream's own size wins."""
    expected = (int(os.environ.get("FASTGRAB_FIXTURE_WIDTH", 320)),
                int(os.environ.get("FASTGRAB_FIXTURE_HEIGHT", 240)))
    with PipeWireVideoReader(node_id) as reader:
        frame = reader.read()
        assert reader.size == expected
    assert frame.shape[:2] == (expected[1], expected[0])


def test_the_channel_order_is_bgra(node_id):
    """The fixture paints solid red, so red must land in byte 2.

    A reader that passed the buffer through as RGBA would put it in byte
    0, and the frame would still be the right shape and dtype.
    """
    with PipeWireVideoReader(node_id) as reader:
        frame = reader.read()
    blue, green, red = frame[..., 0].mean(), frame[..., 1].mean(), frame[..., 2].mean()
    assert red > 200, "expected the fixture's red in byte 2, got %.1f" % red
    assert blue < 50 and green < 50, (
        "channels look swapped: B=%.1f G=%.1f R=%.1f" % (blue, green, red)
    )


def test_reading_twice_reuses_one_pipeline(node_id):
    """Renegotiating per frame is where first-frame failures come from."""
    with PipeWireVideoReader(node_id) as reader:
        first = reader.read()
        pipeline = reader._pipeline
        second = reader.read()
        assert reader._pipeline is pipeline
    assert first.shape == second.shape


def test_an_unknown_node_id_captures_the_wrong_source(node_id):
    """A known gap, pinned rather than papered over.

    ``pipewiresrc path=<id>`` does bind to the node it is asked for --
    measured with two sources on one graph, a red one and a blue one,
    and each id returned its own colour. What it does not do is fail
    when the id is *absent*: PipeWire falls back to any other source on
    the graph and hands back its frames instead. So reading node 99999
    here returns the fixture's red, which for a screen-capture backend
    is the silent wrong-screen bug.

    Two cures were measured and both rejected. ``autoconnect=false``
    stops the stream connecting to anything at all -- valid node ids
    included -- and then deadlocks ``close()``, because pipewiresrc's
    streaming task never finishes and ``set_state(NULL)`` waits on it
    forever. ``node.dont-reconnect`` governs reconnection after a
    target disappears, not the initial fallback, and changes nothing.

    The gap is narrow in practice: the node id arrives from the portal
    over the portal's own connection, where the only visible nodes are
    the ones it shared. It is still real, so it is asserted rather than
    hidden. If pipewiresrc ever starts refusing, this test fails loudly
    -- at which point tighten it into that refusal and drop the caveat
    from PipeWireVideoReader.start().
    """
    assert node_id != 99999, "pick a bogus id the fixture cannot hold"
    reader = PipeWireVideoReader(99999, timeout=5.0)
    try:
        try:
            frame = reader.read()
        except RuntimeError as exc:
            pytest.fail(
                "an unknown node id now fails ({}) -- good, but this "
                "test and the comment in start() both claim it does "
                "not. Tighten both.".format(exc)
            )
    finally:
        reader.close()
    assert frame[..., 2].mean() > 200, (
        "node 99999 neither refused nor returned the fixture's red: "
        "the fallback behaviour has changed and the comment in "
        "PipeWireVideoReader.start() is now wrong"
    )


def test_close_is_idempotent(node_id):
    reader = PipeWireVideoReader(node_id)
    reader.read()
    reader.close()
    reader.close()
    assert reader._pipeline is None
