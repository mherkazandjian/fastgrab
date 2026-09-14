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
import tempfile
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


class PortalConfigError(ValueError):
    """A fastgrab setting is wrong, as opposed to a backend being absent.

    Its own type because auto-detection has to tell the two apart, and
    ValueError alone cannot: ``gi.require_version()`` raises ValueError
    for a missing typelib, so keying on that would turn "GStreamer's
    introspection data is not installed" -- an ordinary partial install,
    and a perfectly good reason to fall back to XWayland -- into a hard
    failure of Screenshot().

    Still a ValueError, so callers that already catch one keep working.
    """


def _persist_mode(explicit=None):
    """Resolve the persist mode from the argument, then the environment."""
    value = explicit if explicit is not None else os.environ.get(PERSIST_ENV)
    if value is None:
        return PERSIST_MODES["transient"]
    key = str(value).strip().lower()
    if key not in PERSIST_MODES:
        raise PortalConfigError(
            "{} must be one of {}, got {!r}".format(
                PERSIST_ENV, ", ".join(sorted(PERSIST_MODES)), value
            )
        )
    return PERSIST_MODES[key]


#: Tokens seen in this process, so a second Screenshot in one program
#: does not re-prompt even when nothing is written to disk.
#:
#: Keyed by persist mode, because the modes must not share. A restore
#: *consumes* its token and the portal issues a replacement; a transient
#: backend keeps that replacement only in memory, so if it were allowed
#: to pick up the persistent token from disk it would spend it and leave
#: the stored one dead -- and the next process, the one persistence
#: exists for, would be prompted after all.
#:
#: The retained bus connection lives under its own key: it belongs to
#: the process, not to a mode. See _save_token.
_PROCESS_TOKEN = {}

TOKEN_ENV = "FASTGRAB_PORTAL_TOKEN_FILE"


def _token_path():
    """Where a ``persistent`` restore token is kept.

    Under ``$XDG_STATE_HOME`` (state, not config: it is regenerated on
    demand and rotates on every use, and losing it costs one prompt).
    """
    override = os.environ.get(TOKEN_ENV)
    if override:
        return override
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "fastgrab", "portal-restore-token")


def _read_token():
    try:
        with open(_token_path()) as handle:
            return handle.read().strip() or None
    except (IOError, OSError):
        # No token, an unreadable one, or no home directory. All of them
        # mean the same thing to the caller: ask the user.
        return None


def _write_token(token):
    """Store a token for a later process, readable only by its owner.

    This is a capability: it lets fastgrab reopen the screen share
    without asking again, which is exactly what ``persistent`` was
    chosen for. It is never written unless that mode was asked for.
    """
    path = _token_path()
    temporary = None
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        # Written to a private temporary file and renamed over the top.
        # Not os.open(..., O_CREAT, 0o600): that mode applies only when
        # the file is *created*, so a token file that already exists
        # group- or world-readable -- copied between machines, restored
        # from a backup, left by an older version -- would silently keep
        # those permissions and be handed the rotated capability
        # anyway. Replacing the inode gives the new file's mode, every
        # time, and never leaves a half-written token behind.
        fd, temporary = tempfile.mkstemp(
            dir=directory, prefix=".portal-restore-token-")
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, token.encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(temporary, path)
        temporary = None
    except (IOError, OSError):
        # Not being able to remember is a lost convenience, never a
        # failed capture.
        pass
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except (IOError, OSError):
                pass


def _forget_token(mode=None):
    """Drop a rejected token. One mode's, or every mode's.

    The connection goes too: it was retained only to keep a remembered
    token usable, and there is no longer one to keep.
    """
    if mode is None:
        for key in list(_PROCESS_TOKEN):
            if key != "connection":
                _PROCESS_TOKEN.pop(key, None)
    else:
        _PROCESS_TOKEN.pop(mode, None)
    _PROCESS_TOKEN.pop("connection", None)
    # Only the mode that owns the file. The stored token belongs to
    # persistent mode alone, so deleting it because a *transient* token
    # was rejected would throw away a perfectly good saved approval over
    # an unrelated failure -- and the transient retry keeps its
    # replacement in memory only, so the next process would be prompted
    # for nothing.
    if mode is None or mode == PERSIST_MODES["persistent"]:
        try:
            os.remove(_token_path())
        except (IOError, OSError):
            pass


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
        from ._portal_dbus import (
            PortalCancelled, ScreenCastSession, _require_gio,
            probe_screencast)
        from ._pipewire import PipeWireVideoReader, _require_gst
        self._session_class = ScreenCastSession
        self._reader_class = PipeWireVideoReader
        self._cancelled = PortalCancelled
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
        # And that a portal is actually there. Having the libraries says
        # nothing about the desktop: a Wayland session with no ScreenCast
        # implementation would otherwise take this backend instead of the
        # XWayland fallback that works. A property read, no session.
        probe_screencast()
        # Not __del__: the same reason as the wlr and windows backends.
        # atexit=False because tearing a GStreamer pipeline down during
        # interpreter shutdown is not worth the risk -- a leaked
        # pipeline at exit costs nothing, a segfault costs the run.
        self._finalizer = weakref.finalize(
            self, _release_portal, self.__dict__)
        self._finalizer.atexit = False

    # -------- session lifetime --------

    def _load_token(self):
        """The restore token to offer, if any.

        ``none`` never offers one -- that is what asking every time
        means. ``transient`` remembers only for this process, so a
        program that builds several Screenshots is asked once.
        ``persistent`` also looks on disk, which is what survives a
        reboot.
        """
        if self._persist == PERSIST_MODES["none"]:
            return None
        token = _PROCESS_TOKEN.get(self._persist)
        if token is None and self._persist == PERSIST_MODES["persistent"]:
            token = _read_token()
        return token

    def _save_token(self, token, session=None):
        if not token or self._persist == PERSIST_MODES["none"]:
            return
        _PROCESS_TOKEN[self._persist] = token
        # The bus connection is kept with it, and deliberately not the
        # session. A transient permission belongs to the D-Bus *client*,
        # and Gio's shared session bus does not outlive its last
        # reference: measured, dropping it and asking again returns a
        # connection with a new unique name (:1.0 -> :1.1). In a script
        # whose only Gio user is fastgrab, releasing the last Screenshot
        # would therefore leave a remembered token that the desktop no
        # longer recognises, and the next one would prompt after all.
        # Holding the session instead would keep the *capture* alive,
        # which is not wanted; the connection alone is enough.
        if session is not None and session.connection is not None:
            _PROCESS_TOKEN["connection"] = session.connection
        if self._persist == PERSIST_MODES["persistent"]:
            _write_token(token)

    def _open_session(self, use_token=True):
        """Open one portal session, reusing consent where allowed.

        The portal rotates the token on every Start, so the one that
        comes back replaces the one that went in.
        """
        token = self._load_token() if use_token else None
        session = self._session_class(
            app_id="fastgrab", persist_mode=self._persist,
            timeout=self._timeout, restore_token=token)
        try:
            session.open()
        except BaseException as exc:
            # CreateSession can succeed and Start still fail, time out or
            # be interrupted. The session never reaches self._session in
            # that case, so neither close() nor the finalizer could
            # release it, and on a long-lived bus connection the
            # half-open session and its pending chooser outlive the
            # failed capture -- with every retry adding another.
            session.close()
            if token is None or not isinstance(exc, Exception):
                raise
            if isinstance(exc, self._cancelled):
                # They said no. Asking again immediately is not a retry,
                # it is nagging.
                raise
            # A stored token the desktop no longer honours would
            # otherwise wedge capture until somebody found and deleted
            # the file. Drop it and ask properly, once.
            _forget_token(self._persist)
            # use_token=False rather than trusting the deletion.
            # _forget_token() cannot guarantee the file is gone -- an
            # unwritable directory makes the unlink fail and it
            # deliberately swallows that -- and re-reading the same
            # rejected token would open a session per attempt until
            # RecursionError. This bounds it at one retry whatever
            # happened on disk.
            return self._open_session(use_token=False)
        self._save_token(session.restore_token, session)
        return session

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
        reader = None
        try:
            reader = self._reader_class(self._session.node_id, fd=fd)
            reader.start()
        except BaseException:
            # The reader too, not just the descriptor. start() can fail
            # or be interrupted *after* the pipeline reaches PLAYING,
            # and this one never reaches self._reader -- so neither
            # close() nor the finalizer would ever see it, and it would
            # keep running for the life of the process.
            if reader is not None:
                reader.close()
            os.close(fd)
            raise
        self._fd = fd
        return reader

    def _ensure_stream(self):
        """Open the portal session and the reader, once."""
        if self._reader is not None:
            return self._reader
        if self._session is None:
            self._session = self._open_session()
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
