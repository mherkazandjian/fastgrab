"""xdg-desktop-portal + PipeWire capture, for GNOME and KDE Wayland.

The backend the ``wlr`` one cannot be: GNOME and KDE do not implement
``wlr-screencopy-v1``, so on those desktops the portal is the only way
in. It asks the user's permission -- that is the point of it -- and the
approval belongs to the session this backend holds open, so one
:class:`fastgrab.screenshot.Screenshot` means one prompt.

Opt-in via ``pip install fastgrab[portal]``. That brings PyGObject;
GStreamer and its introspection data are system packages pip cannot
install (``gir1.2-gst-plugins-base-1.0``, ``gstreamer1.0-pipewire`` on
Debian/Ubuntu). Nothing here is imported by a default install.

Split in two, because the halves fail for different reasons and are
testable in different ways:

* :mod:`fastgrab.backends._portal_dbus` -- the ScreenCast handshake
  (CreateSession, SelectSources, Start, OpenPipeWireRemote).
* :mod:`fastgrab.backends._pipewire` -- reading frames off the node id
  that handshake returns.

How this is tested without a GPU
--------------------------------

``docker compose run --rm test-portal``. The PipeWire half runs against
a synthetic ``videotestsrc`` node with no compositor at all. The portal
half runs against the *real* ``xdg-desktop-portal`` frontend, with
``tests/portal/fake_impl_screencast.py`` standing in for the desktop's
``org.freedesktop.impl.portal.ScreenCast`` backend -- which is the
component GNOME and KDE each supply, and the component that shows the
consent dialog. So the wire protocol, the Request/Response signal
dance, the unix-FD list handling and the session lifetime are all
exercised for real; only the human clicking "Share" is faked.

What is *not* covered is the real xdg-desktop-portal-wlr path, and the
reason is worth keeping because it is easy to rediscover the hard way.

What actually blocks it
-----------------------

That run only succeeded because the container was handed a DMA-BUF
capable GPU (``docker run --device /dev/dri`` with
``WLR_RENDER_DRM_DEVICE=/dev/dri/renderD129``, an amdgpu render node).
Without one, wlroots falls back to the pixman software renderer, never
emits the ``zwlr_screencopy_frame_v1.linux_dmabuf`` event, and xdpw
0.7.1 refuses to start the cast — ``src/screencast/screencast.c``,
``start_screencast()``::

    if (cast->screencopy_frame_info[WL_SHM].format == DRM_FORMAT_INVALID ||
            (cast->ctx->state->screencast_version >= 3 &&
             cast->screencopy_frame_info[DMABUF].format == DRM_FORMAT_INVALID)) {
        logprint(INFO, "wlroots: unable to receive a valid format from wlr_screencopy");
        return -1;
    }

Once the compositor advertises ``zwlr_screencopy_manager_v1`` v3 — cage
does — a SHM format alone is not enough. ``Start`` then comes back with
response code 2 and the session is torn down immediately. This is
upstream issue emersion/xdg-desktop-portal-wlr#289, "pixman: not
supported as a renderer with screencast_version >= 3", still open, and
reported from exactly where we hit it: a headless docker container.
xdpw reads exactly one environment variable (``XDPW_PERSIST_MODE``) and
exposes no config key to prefer, or fall back to, SHM.

There is no software escape hatch. wlroots will not build a GL renderer
without a DRM fd at all::

    [ERROR] [render/wlr_renderer.c:98] drmGetDevices2 failed: ...
    [ERROR] [render/wlr_renderer.c:192] Cannot create GLES2 renderer: no DRM FD available

so "just use llvmpipe" is not on the table — pixman is the only
device-free renderer, and pixman is the thing xdpw rejects.

That leaves the test un-runnable where it would have to run. GitHub's
hosted runners have no GPU and no ``/dev/dri``, so the CI job could not
pass. And a ``docker compose run --rm test-portal`` service would need
``--device /dev/dri`` *plus* a Mesa-supported GPU on the contributor's
machine: on the very host this spike ran, the NVIDIA render node
(renderD128) failed with "DRI2: failed to create screen" and only the
AMD one (renderD129) worked. "Does it work in ``docker compose run
--rm test``?" would stop having a reliable answer, which is the one
question this project most needs to keep.

Why PyGObject, and not the libraries the extra used to name
-----------------------------------------------------------

Both alternatives were tried and neither works:

* ``pipewire-python`` cannot do the job. It is a subprocess wrapper
  around the ``pw-cat`` / ``pw-play`` / ``pw-record`` CLIs; its entire
  public API is ``playback`` / ``record`` / ``get_list_targets``, it is
  audio-only, and it can neither target a node id, nor accept the
  portal's fd, nor hand back a raw video buffer. It also installs as
  ``pipewire_python``, so the ``import pipewire`` guard this file used
  to run raised ImportError even when the extra *was* installed.

* ``dbus-next`` 0.2.3 cannot introspect the portal at all::

      dbus_next.errors.InvalidMemberNameError:
          invalid member name: power-saver-enabled

  xdg-desktop-portal >= 1.18 exposes a hyphenated property name, and
  dbus-next rejects it (correctly, per the D-Bus spec) while parsing
  the whole introspection document -- so a single unrelated interface
  takes down ``bus.introspect()``. Workable by hand-writing the
  introspection XML, but dbus-next has been unmaintained since 2021 and
  this is the first thing it gets wrong.

PyGObject brings Gio (D-Bus, including the unix-FD-list calls the
portal needs for OpenPipeWireRemote) and GStreamer (``pipewiresrc``) in
one dependency, and both are maintained. The old ``[wayland-portal]``
extra pulled in dbus-next; it is now an alias for ``[portal]`` so the
documented command keeps working.

Notes for anyone changing this
------------------------------

* ``OpenPipeWireRemote`` returns D-Bus type ``h``, which on the wire is
  an *index into the message's FD list*, not a descriptor. Treating the
  integer as an fd reads from whatever unrelated file holds that
  number. It is also single-use per consumer connection.
* The node id is only meaningful on the portal's own connection.
  ``pipewiresrc`` defaults to ``autoconnect=true``, which makes a
  wrong or stale node id succeed against some other source instead of
  failing -- the silent wrong-screen bug. Both are pinned by tests.
* Portal ``Request`` objects have a predictable path derived from the
  handle token, so subscribe to the ``Response`` signal *before*
  issuing the call, or lose the race.
* Probe ``interface_version`` and degrade options it does not
  advertise. The spike saw version 5, ``AvailableSourceTypes=1``
  (monitor only) and ``AvailableCursorModes=3`` from xdpw 0.7.1;
  xdg-desktop-portal-gnome and -gtk differ on ``cursor_mode`` and
  ``persist_mode``.
"""
import os
import weakref

from .base import BaseBackend


#: How a session asks the desktop to remember consent. Selectable so a
#: user can decide whether fastgrab is allowed to keep an approval, which
#: is a privacy question and not ours to answer for them.
#:
#:   none        ask every time a session starts (the portal default)
#:   transient   remember for as long as the desktop session lasts
#:   persistent  remember across reboots, via a token the desktop stores
PERSIST_MODES = {"none": 0, "transient": 1, "persistent": 2}
PERSIST_ENV = "FASTGRAB_PORTAL_PERSIST"


def _persist_mode(explicit=None):
    """Resolve the persist mode from the argument, then the environment."""
    value = explicit if explicit is not None else os.environ.get(PERSIST_ENV)
    if value is None:
        return PERSIST_MODES["transient"]
    key = str(value).strip().lower()
    if key not in PERSIST_MODES:
        raise ValueError(
            "{} must be one of {}, got {!r}".format(
                PERSIST_ENV, ", ".join(sorted(PERSIST_MODES)), value
            )
        )
    return PERSIST_MODES[key]


def _release_portal(state):
    """Tear down a discarded backend's stream and session.

    Takes the instance ``__dict__`` rather than the instance: a
    finalizer that referenced the backend would keep it alive and never
    run. Reading the dict also means it sees whatever the handles are
    *now*, not what they were at construction.
    """
    try:
        reader = state.get("_reader")
        if reader is not None:
            reader.close()
    finally:
        # In a finally: this is the descriptor the finalizer exists to
        # reclaim, and a reader that threw on the way down must not
        # take it with it.
        fd = state.get("_fd")
        if fd is not None:
            os.close(fd)
        session = state.get("_session")
        if session is not None:
            session.close()
        state["_reader"] = state["_session"] = state["_fd"] = None


class PortalBackend(BaseBackend):
    """Capture through xdg-desktop-portal ScreenCast and PipeWire.

    **Consent.** Starting a portal session is what makes the desktop ask
    the user, and the approval belongs to that session. This backend
    therefore keeps one session open for its whole life, so a program
    that reuses a :class:`fastgrab.screenshot.Screenshot` is asked once.
    A program that builds one per frame is asked per frame — which is
    what the library's own two-line form does, so on GNOME and KDE reuse
    is not a micro-optimisation but the difference between one prompt and
    hundreds.

    Nothing is asked at construction. Autodetection builds backends
    speculatively, and a chooser dialog appearing because a program
    called ``Screenshot()`` would be indefensible; the session opens on
    first use.

    ``persist`` (or ``$FASTGRAB_PORTAL_PERSIST``) chooses whether the
    desktop may remember the approval — see :data:`PERSIST_MODES`.
    """

    def __init__(self, persist=None, timeout=60.0):
        self._persist = _persist_mode(persist)
        self._timeout = float(timeout)
        self._session = None
        self._reader = None
        self._fd = None
        # Import here, not at module scope: the dependencies live behind
        # the [portal] extra and importing this module must stay cheap
        # and safe for a default install.
        from ._portal_dbus import ScreenCastSession, _require_gio
        from ._pipewire import PipeWireVideoReader, _require_gst
        self._session_class = ScreenCastSession
        self._reader_class = PipeWireVideoReader
        # Probe the optional dependencies now rather than at first
        # capture. _autodetect() decides which backend to use by
        # constructing one and catching the failure, so a constructor
        # that succeeds without PyGObject and GStreamer claims every
        # GNOME and KDE session for a backend that cannot work -- and
        # the XWayland fallback that would have worked never runs. The
        # error then surfaces from capture(), outside the handler that
        # exists to catch exactly this.
        #
        # Neither probe talks to the portal, so neither prompts anyone;
        # consent still waits for the first capture.
        _require_gio()
        _require_gst()
        # Not __del__: the same reason as the wlr and windows backends.
        # atexit=False because tearing a GStreamer pipeline down during
        # interpreter shutdown is not worth the risk -- a leaked
        # pipeline at exit costs nothing, a segfault costs the run.
        self._finalizer = weakref.finalize(
            self, _release_portal, self.__dict__)
        self._finalizer.atexit = False

    # -------- session lifetime --------

    def _open_reader(self):
        """Build a reader bound to the portal's own PipeWire connection.

        The node id is only meaningful on the connection the portal
        hands back: OpenPipeWireRemote returns a pre-authenticated
        socket, and the same number on the ambient socket is a
        different node, or somebody else's. It happens to work when
        both are the same daemon, which is exactly why leaving it out
        survives testing and then captures the wrong screen.

        The descriptor is single-use per consumer connection, so
        refresh() comes back through here for a new one instead of
        holding on to this one.
        """
        fd = self._session.open_pipewire_remote()
        try:
            reader = self._reader_class(self._session.node_id, fd=fd)
            reader.start()
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return reader

    def _ensure_stream(self):
        """Open the portal session and the reader, once."""
        if self._reader is not None:
            return self._reader
        if self._session is None:
            session = self._session_class(
                app_id="fastgrab", persist_mode=self._persist,
                timeout=self._timeout)
            session.open()
            # Stored before the stream is built, so a failure below
            # leaves the session reachable by close() rather than open
            # and orphaned. It is deliberately not closed here: consent
            # is attached to it, and throwing it away would make a
            # retry prompt the user a second time.
            self._session = session
        self._reader = self._open_reader()
        return self._reader

    def _close_stream(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def close(self):
        self._close_stream()
        if self._session is not None:
            self._session.close()
            self._session = None
        self._closed = True
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()

    def refresh(self):
        """Drop the stream so the next capture renegotiates.

        Not the session: re-opening that would ask for consent again.
        """
        self._close_stream()

    # -------- BaseBackend --------

    def resolution(self):
        """The stream's own size, from the negotiated video caps.

        Not the portal's ``size`` property: that is in compositor
        coordinates and need not match the pixels delivered.
        """
        self._check_open()
        return tuple(self._ensure_stream().size)

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        self._check_open()
        height, width = img.shape[:2]
        frame = self._ensure_stream().read()
        stream_h, stream_w = frame.shape[:2]
        if (x < 0 or y < 0 or x + width > stream_w
                or y + height > stream_h):
            raise ValueError(
                "region ({}, {}, {}, {}) is outside the {}x{} portal "
                "stream".format(x, y, width, height, stream_w, stream_h)
            )
        # Cropped here rather than asked of the compositor: the portal
        # hands back a whole monitor and has no sub-region concept.
        img[:] = frame[y:y + height, x:x + width]
