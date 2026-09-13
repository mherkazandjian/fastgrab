"""The xdg-desktop-portal ScreenCast handshake.

Separated from frame reading (:mod:`fastgrab.backends._pipewire`) because
the two fail for entirely different reasons and can be tested in
entirely different ways: this half needs a portal on a session bus, that
half needs a PipeWire graph.

The conversation is CreateSession -> SelectSources -> Start, each of
which returns immediately and delivers its real answer later on a
``Response`` signal against a Request object. Start is where a desktop
environment shows its chooser, so it is also where the user can say no.
"""
import os


def _require_gio():
    try:
        import gi
    except ImportError as exc:
        raise RuntimeError(
            "the portal backend needs PyGObject: pip install "
            "'fastgrab[portal]'"
        ) from exc
    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib
    return Gio, GLib


PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"


class PortalCancelled(RuntimeError):
    """The user declined, or dismissed the chooser.

    Distinct from every other failure on purpose. Falling back to
    XWayland after someone has actively said no would capture a screen
    they just refused to share.
    """


class PortalUnavailable(RuntimeError):
    """No portal, or no ScreenCast implementation behind it."""


def probe_screencast(timeout=5.0):
    """Confirm a working ScreenCast portal is reachable, and nothing more.

    Reads the interface's ``version`` property. That is enough to tell
    the two cases apart, measured on a bare session bus: the frontend is
    D-Bus-activatable and does start, but with no desktop
    ``impl.portal.ScreenCast`` behind it it does not export the
    interface at all, and the read comes back ``No such interface``.

    This matters because auto-detection decides by construction. A
    Wayland session with PyGObject and GStreamer installed but no portal
    -- a wlroots desktop that never installed one, say -- would
    otherwise be handed this backend while its XWayland worked fine, and
    would only find out at capture time.

    No session is created and no chooser is shown, so nobody is asked
    anything. Activating the frontend is the one side effect, and that
    is an ordinary session service that any screen-share would start.
    """
    Gio, GLib = _require_gio()
    try:
        conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except Exception as exc:
        raise PortalUnavailable(
            "no session bus to reach the portal on: {}".format(exc)
        ) from exc
    try:
        reply = conn.call_sync(
            PORTAL_BUS, PORTAL_PATH, "org.freedesktop.DBus.Properties", "Get",
            GLib.Variant("(ss)", (SCREENCAST_IFACE, "version")),
            GLib.VariantType("(v)"), Gio.DBusCallFlags.NONE,
            int(timeout * 1000), None)
    except Exception as exc:
        raise PortalUnavailable(
            "no xdg-desktop-portal ScreenCast implementation on the "
            "session bus ({}). On GNOME or KDE install "
            "xdg-desktop-portal-gnome or -kde; on wlroots, "
            "xdg-desktop-portal-wlr.".format(exc)
        ) from exc
    return reply.unpack()[0]


class ScreenCastSession:
    """One portal session, held open across captures.

    Held open because consent is attached to it: a desktop environment
    prompts when a session starts, so a session per screenshot is a
    prompt per screenshot. Callers that want one approval keep one
    session alive.
    """

    def __init__(self, app_id="", multiple=False, cursor_mode=1,
                 timeout=60.0, persist_mode=0, restore_token=None):
        self._gio, self._glib = _require_gio()
        self.app_id = app_id
        self.multiple = multiple
        self.cursor_mode = cursor_mode
        self.timeout = timeout
        # 0 ask every time, 1 remember for this desktop session, 2 across
        # reboots. The desktop decides whether it honours the request.
        self.persist_mode = int(persist_mode)
        self.restore_token = restore_token
        self._conn = None
        self._session_handle = None
        self.streams = []

    # -------- plumbing --------

    def _connect(self):
        Gio = self._gio
        if self._conn is None:
            try:
                self._conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            except Exception as exc:
                raise PortalUnavailable(
                    "no session bus to reach the portal on: {}".format(exc)
                ) from exc
        return self._conn

    def _unique_token(self, prefix):
        # The request path is predictable from the token, which is what
        # lets a caller subscribe *before* issuing the call. Without
        # that, a portal that answers quickly can emit Response before
        # the subscription exists and the call waits forever.
        return "fastgrab_{}_{}".format(prefix, os.getpid() and id(self) & 0xffffff)

    def _request_path(self, token):
        sender = self._connect().get_unique_name()[1:].replace(".", "_")
        return "/org/freedesktop/portal/desktop/request/{}/{}".format(
            sender, token
        )

    def _call_with_response(self, method, args_builder, token_prefix):
        """Call a portal method and wait for its Response signal."""
        Gio, GLib = self._gio, self._glib
        conn = self._connect()
        token = self._unique_token(token_prefix)
        path = self._request_path(token)

        result = {}
        loop = GLib.MainLoop()

        def on_response(_c, _sender, _path, _iface, _signal, params):
            result["response"], result["results"] = params.unpack()
            loop.quit()

        subscription = conn.signal_subscribe(
            PORTAL_BUS, REQUEST_IFACE, "Response", path, None,
            Gio.DBusSignalFlags.NONE, on_response)

        fired = []

        def give_up():
            result.setdefault("timeout", True)
            fired.append(True)
            loop.quit()
            return False

        timer = GLib.timeout_add(int(self.timeout * 1000), give_up)
        try:
            conn.call_sync(
                PORTAL_BUS, PORTAL_PATH, SCREENCAST_IFACE, method,
                args_builder(token), None, Gio.DBusCallFlags.NONE, -1, None)
            loop.run()
        finally:
            # Only if it has not already fired: removing a spent source
            # warns, and the warning is noise on the very path that is
            # already reporting a real failure.
            if not fired:
                GLib.source_remove(timer)
            conn.signal_unsubscribe(subscription)

        if result.get("timeout"):
            raise PortalUnavailable(
                "the portal never answered {} within {}s".format(
                    method, self.timeout)
            )
        code = result.get("response")
        if code == 1:
            raise PortalCancelled(
                "the screen-share request was declined at {}".format(method)
            )
        if code != 0:
            raise PortalUnavailable(
                "the portal refused {} with response {}".format(method, code)
            )
        return result.get("results") or {}

    # -------- the conversation --------

    def _select_options(self, token):
        GLib = self._glib
        options = {
            "handle_token": GLib.Variant("s", token),
            "types": GLib.Variant("u", 1),          # MONITOR
            "multiple": GLib.Variant("b", self.multiple),
            "cursor_mode": GLib.Variant("u", self.cursor_mode),
            "persist_mode": GLib.Variant("u", self.persist_mode),
        }
        if self.restore_token:
            # Single-use: the portal replaces it on every successful
            # restore, so a caller that keeps the old one is asked again.
            options["restore_token"] = GLib.Variant("s", self.restore_token)
        return options

    def open(self):
        """Run CreateSession, SelectSources and Start; return the streams."""
        GLib = self._glib
        if self._session_handle is not None:
            return self.streams

        session_token = self._unique_token("session")
        results = self._call_with_response(
            "CreateSession",
            lambda token: GLib.Variant("(a{sv})", ({
                "handle_token": GLib.Variant("s", token),
                "session_handle_token": GLib.Variant("s", session_token),
            },)),
            "create")
        self._session_handle = results.get("session_handle")
        if not self._session_handle:
            raise PortalUnavailable("the portal returned no session handle")

        self._call_with_response(
            "SelectSources",
            lambda token: GLib.Variant("(oa{sv})", (self._session_handle,
                                                     self._select_options(token))),
            "select")

        results = self._call_with_response(
            "Start",
            lambda token: GLib.Variant("(osa{sv})", (self._session_handle, "", {
                "handle_token": GLib.Variant("s", token),
            })),
            "start")
        # Replaced on every successful restore, so it has to be read back
        # rather than assumed to be the one that was sent.
        self.restore_token = results.get("restore_token") or self.restore_token
        self.streams = list(results.get("streams") or [])
        if not self.streams:
            raise PortalUnavailable(
                "the portal approved the request but offered no streams"
            )
        return self.streams

    @property
    def node_id(self):
        """The PipeWire node id of the first approved stream."""
        if not self.streams:
            self.open()
        return self.streams[0][0]

    def open_pipewire_remote(self):
        """The file descriptor for the PipeWire connection to use.

        Returned as a *unix FD list* entry: D-Bus type ``h`` on the wire
        is an index into the message's descriptor list, not a descriptor
        itself. Treating the integer as an fd is the classic mistake
        here and would read from whatever unrelated file happened to
        occupy that number.
        """
        Gio, GLib = self._gio, self._glib
        conn = self._connect()
        if self._session_handle is None:
            self.open()
        reply, fd_list = conn.call_with_unix_fd_list_sync(
            PORTAL_BUS, PORTAL_PATH, SCREENCAST_IFACE, "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (self._session_handle, {})),
            GLib.VariantType("(h)"), Gio.DBusCallFlags.NONE, -1, None, None)
        index = reply.unpack()[0]
        return fd_list.get(index)

    def close(self):
        Gio, GLib = self._gio, self._glib
        if self._session_handle and self._conn is not None:
            try:
                self._conn.call_sync(
                    PORTAL_BUS, self._session_handle, SESSION_IFACE, "Close",
                    None, None, Gio.DBusCallFlags.NONE, 5000, None)
            except Exception:
                # The portal may already have dropped it; closing twice
                # must not be an error for the caller.
                pass
        self._session_handle = None
        self.streams = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
