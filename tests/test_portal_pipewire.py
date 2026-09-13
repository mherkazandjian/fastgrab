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


def test_an_idle_stream_still_answers_with_the_last_frame(node_id):
    """A screencast emits on *damage*, not on a clock.

    Point one at a desktop where nothing moves and it delivers no
    buffers at all. Requiring a fresh one per capture would make a
    still screen indistinguishable from a dead stream, and would fail
    the first capture too: resolution() consumes the opening frame
    through size, leaving nothing for the screenshot that follows.

    This fixture emits continuously at 10fps, which hides the case
    entirely -- hence the stand-in sink that stops delivering.
    """
    reader = PipeWireVideoReader(node_id, timeout=2.0)
    try:
        first = reader.read()
        reader._sink.try_pull_sample = lambda _ns: None
        second = reader.read()
    finally:
        reader.close()
    assert numpy.array_equal(first, second)


def test_a_stream_that_has_never_delivered_still_fails(node_id):
    """The control on the test above.

    Serving the last frame is only right once there *is* one. A stream
    that has produced nothing is a broken stream and has to say so,
    rather than quietly handing back an empty array.
    """
    reader = PipeWireVideoReader(node_id, timeout=1.0)
    reader.start()
    reader._sink.try_pull_sample = lambda _ns: None
    try:
        with pytest.raises(RuntimeError, match="no frame"):
            reader.read()
    finally:
        reader.close()


def test_close_drops_the_cached_frame(node_id):
    """A reopened reader must not answer with the old stream's frame.

    The reopen itself cannot be exercised against this fixture -- its
    provider is `pipewiresink mode=provide`, which exits as soon as its
    consumer disconnects, so by the time a second start() runs the node
    is off the graph. The invariant that carries the risk is testable
    though: the cache goes away with the pipeline, so there is nothing
    stale left to hand back.
    """
    reader = PipeWireVideoReader(node_id, timeout=1.0)
    reader.read()
    assert reader._last is not None
    reader.close()
    assert reader._last is None, (
        "a closed reader kept the last frame, so reopening it would "
        "answer with the previous stream's picture"
    )


def test_a_padded_row_stride_is_honoured(node_id):
    """Rows are not always width*4 apart.

    A producer may pad each row out to a hardware-friendly stride, or
    start the plane at an offset, and videoconvert passes such a buffer
    straight through when it is already BGRx. Reshaping on width alone
    reads the padding as pixels and shears the image further on every
    row -- so the deliberately padded buffer here is built by hand,
    because whether this fixture ever produces one is not something the
    test should depend on.
    """
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo

    width, height = 7, 3
    stride = width * 4 + 20          # 20 bytes of padding per row
    offset = 8                       # and the plane does not start at 0
    raw = bytearray(offset + stride * height)
    for y in range(height):
        for x in range(width):
            at = offset + y * stride + x * 4
            raw[at:at + 4] = bytes((x, y, 200, 255))
    # The padding is filled with a value no real pixel here carries, so
    # leaking any of it into the result is unmistakable.
    for y in range(height):
        at = offset + y * stride + width * 4
        raw[at:at + (stride - width * 4)] = b"\xee" * (stride - width * 4)

    buffer = Gst.Buffer.new_wrapped(bytes(raw))
    GstVideo.buffer_add_video_meta_full(
        buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.BGRX,
        # Fixed-width arrays: GStreamer wants GST_VIDEO_MAX_PLANES (4)
        # entries whatever the format's real plane count is.
        width, height, 1, [offset, 0, 0, 0], [stride, 0, 0, 0])
    caps = Gst.Caps.from_string(
        "video/x-raw,format=BGRx,width=%d,height=%d" % (width, height))
    sample = Gst.Sample.new(buffer, caps, None, None)

    reader = PipeWireVideoReader(node_id, timeout=2.0)
    try:
        reader.start()
        reader._sink.try_pull_sample = lambda _ns: sample
        frame = reader.read()
    finally:
        reader.close()

    assert frame.shape == (height, width, 4)
    assert not (frame == 0xEE).any(), "row padding leaked into the image"
    for y in range(height):
        for x in range(width):
            assert tuple(frame[y, x]) == (x, y, 200, 255), (
                "pixel (%d, %d) came back as %s -- the rows are sheared"
                % (x, y, tuple(frame[y, x]))
            )


def test_close_is_idempotent(node_id):
    reader = PipeWireVideoReader(node_id)
    reader.read()
    reader.close()
    reader.close()
    assert reader._pipeline is None
