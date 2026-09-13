"""Read frames from a PipeWire video node into numpy BGRA arrays.

This is the half of the portal backend that does *not* depend on
xdg-desktop-portal. Given a node id — however it was obtained — it hands
back frames, and it can be tested against a synthetic source with no
compositor and no GPU. That matters because the other half is blocked
upstream (see issue #54), and separating them keeps most of the backend
testable in CI rather than none of it.

Imported lazily by :mod:`fastgrab.backends.portal`; the dependencies live
behind the ``portal`` extra and must never be needed by a default install.
"""


def _require_gst():
    """Import GStreamer through PyGObject, or explain what is missing.

    ``GstApp`` has to be required explicitly: without it PyGObject only
    exposes the base ``GstElement`` interface and ``appsink`` has no
    ``try_pull_sample``, which surfaces as a bare AttributeError rather
    than anything that names the cause.
    """
    try:
        import gi
    except ImportError as exc:
        raise RuntimeError(
            "the portal backend needs PyGObject: pip install "
            "'fastgrab[portal]', which also needs GStreamer and its "
            "GObject-introspection data (gir1.2-gst-plugins-base-1.0 and "
            "gstreamer1.0-pipewire on Debian/Ubuntu)"
        ) from exc
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp  # noqa: F401  (GstApp for appsink)
    if not Gst.is_initialized():
        Gst.init(None)
    if Gst.ElementFactory.find("pipewiresrc") is None:
        raise RuntimeError(
            "GStreamer has no 'pipewiresrc' element, so PipeWire frames "
            "cannot be read (install gstreamer1.0-pipewire)"
        )
    return Gst


class PipeWireVideoReader:
    """Pull frames from one PipeWire video node.

    The pipeline is kept open between reads on purpose. Renegotiating per
    frame costs a fresh format handshake each time, and the first frame
    after one is the one most likely to be missing.
    """

    def __init__(self, node_id, timeout=5.0):
        self._gst = _require_gst()
        self.node_id = int(node_id)
        self.timeout = float(timeout)
        self._pipeline = None
        self._sink = None
        self._size = None

    def start(self):
        Gst = self._gst
        if self._pipeline is not None:
            return
        # videoconvert, and no width/height in the caps: the stream's own
        # size is whatever the producer negotiated, and pinning it here
        # fails the state change outright rather than adapting.
        self._pipeline = Gst.parse_launch(
            "pipewiresrc path={} ! videoconvert ! video/x-raw,format=BGRx "
            "! appsink name=out max-buffers=2 drop=true sync=false".format(
                self.node_id
            )
        )
        self._sink = self._pipeline.get_by_name("out")
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            message = self._pipeline.get_bus().timed_pop_filtered(
                int(2 * Gst.SECOND), Gst.MessageType.ERROR
            )
            detail = message.parse_error()[0].message if message else "no error reported"
            self.close()
            raise RuntimeError(
                "could not read PipeWire node {}: {}".format(self.node_id, detail)
            )

    @property
    def size(self):
        """``(width, height)`` of the stream, from the negotiated caps.

        Not from the portal's ``size`` property: that describes compositor
        coordinates and need not match the pixels actually delivered.
        """
        if self._size is None:
            self.read()
        return self._size

    def read(self):
        """Return the most recent frame as a ``(h, w, 4)`` BGRA array."""
        import numpy

        Gst = self._gst
        self.start()
        deadline_ns = int(self.timeout * Gst.SECOND)
        sample = self._sink.try_pull_sample(deadline_ns)
        if sample is None:
            raise RuntimeError(
                "no frame from PipeWire node {} within {}s".format(
                    self.node_id, self.timeout
                )
            )
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        self._size = (width, height)

        buffer = sample.get_buffer()
        ok, info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("could not map the PipeWire buffer")
        try:
            # Copied, not viewed: the mapping is borrowed and is unmapped
            # again below, so a view would dangle.
            flat = numpy.frombuffer(info.data, dtype=numpy.uint8)
            return flat[: height * width * 4].reshape(height, width, 4).copy()
        finally:
            buffer.unmap(info)

    def close(self):
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
            self._sink = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
