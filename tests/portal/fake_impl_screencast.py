"""A fake ``org.freedesktop.impl.portal.ScreenCast`` backend.

The real xdg-desktop-portal *frontend* runs against this, so the client
under test talks to genuine portal code — request objects, response
signals, the session lifetime and the file-descriptor handoff — rather
than to a mock of my own understanding of the protocol.

What is faked is only the part a desktop environment would supply: the
chooser UI and the compositor's PipeWire node. The node id handed back
points at the synthetic source the test started, so a client that
completes the handshake really can read frames from it.
"""
import os
import socket
import sys

import gi
gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib  # noqa: E402

BUS_NAME = "org.freedesktop.impl.portal.desktop.fastgrabfake"
OBJECT_PATH = "/org/freedesktop/portal/desktop"

# Only the members the frontend actually calls on an impl backend.
INTROSPECTION = """
<node>
  <interface name='org.freedesktop.impl.portal.ScreenCast'>
    <property name='version' type='u' access='read'/>
    <property name='AvailableSourceTypes' type='u' access='read'/>
    <property name='AvailableCursorModes' type='u' access='read'/>
    <method name='CreateSession'>
      <arg type='o' name='handle' direction='in'/>
      <arg type='o' name='session_handle' direction='in'/>
      <arg type='s' name='app_id' direction='in'/>
      <arg type='a{sv}' name='options' direction='in'/>
      <arg type='u' name='response' direction='out'/>
      <arg type='a{sv}' name='results' direction='out'/>
    </method>
    <method name='SelectSources'>
      <arg type='o' name='handle' direction='in'/>
      <arg type='o' name='session_handle' direction='in'/>
      <arg type='s' name='app_id' direction='in'/>
      <arg type='a{sv}' name='options' direction='in'/>
      <arg type='u' name='response' direction='out'/>
      <arg type='a{sv}' name='results' direction='out'/>
    </method>
    <method name='Start'>
      <arg type='o' name='handle' direction='in'/>
      <arg type='o' name='session_handle' direction='in'/>
      <arg type='s' name='app_id' direction='in'/>
      <arg type='s' name='parent_window' direction='in'/>
      <arg type='a{sv}' name='options' direction='in'/>
      <arg type='u' name='response' direction='out'/>
      <arg type='a{sv}' name='results' direction='out'/>
    </method>
  </interface>
</node>
"""


class FakeScreenCastImpl:
    def __init__(self, node_id, response=0):
        self.node_id = int(node_id)
        self.response = int(response)
        # Kept alive: the descriptor is duplicated into the message, but
        # closing our end immediately still tears the connection down.
        self._sockets = []

    def _record(self, line):
        """Report to the test what this backend was handed.

        The fake runs as a subprocess, so a file is the channel. Whether
        a restore token arrived is invisible from the client side --
        that is the whole point of the token -- so it has to be observed
        from here.
        """
        state = os.environ.get("FASTGRAB_FAKE_STATE")
        if not state:
            return
        with open(state, "a") as handle:
            handle.write(line + "\n")

    def handle_call(self, _conn, _sender, _path, _iface, method, params, invocation):
        if method == "CreateSession":
            invocation.return_value(GLib.Variant("(ua{sv})", (0, {})))
            return
        if method == "SelectSources":
            options = params.unpack()[3]
            # restore_data, not restore_token, is what a desktop backend
            # sees on a restore: the frontend owns the token database,
            # resolves the client's token itself and hands us back the
            # data we returned from Start. Recording it is the only
            # evidence that a restore actually happened.
            restored = options.get("restore_data")
            self._record("select %s %s %s" % (
                options.get("restore_token") or "-",
                options.get("persist_mode", "-"),
                "restored" if restored else "-"))
            invocation.return_value(GLib.Variant("(ua{sv})", (0, {})))
            return
        if method == "Start":
            if self.response != 0:
                # response 1 is user cancellation, 2 is "something failed".
                invocation.return_value(
                    GLib.Variant("(ua{sv})", (self.response, {})))
                return
            # Plain tuples inside the array, not nested Variants: with a
            # format string GLib builds the children itself, and handing
            # it Variants raises "Expected GLib.Variant, but got tuple"
            # from inside the constructor -- which surfaces as the portal
            # never answering, not as an error the client can see.
            streams = [(self.node_id,
                        {"size": GLib.Variant("(ii)", (320, 240))})]
            results = {"streams": GLib.Variant("a(ua{sv})", streams)}
            # restore_data (suv), which is what the impl contract
            # actually returns: (vendor, version, vendor-defined data).
            # The *frontend* stores it and issues the client a UUID
            # restore_token of its own. Returning a restore_token from
            # here instead creates no restorable permission at all --
            # the string is passed through to the client and then fails
            # the frontend's UUID validation when it is offered back, so
            # the persistence path never actually restores anything.
            if os.environ.get("FASTGRAB_FAKE_RESTORE"):
                results["restore_data"] = GLib.Variant(
                    "(suv)", ("fastgrabfake", 1, GLib.Variant("s", "ok")))
            token = os.environ.get("FASTGRAB_FAKE_TOKEN")
            if token:
                results["restore_token"] = GLib.Variant("s", token)
            invocation.return_value(GLib.Variant("(ua{sv})", (0, results)))
            return
        if method == "OpenPipeWireRemote":
            # A real backend hands back a connection to the compositor's
            # PipeWire instance. Here that is the same daemon the test
            # fixture started, reached through its socket.
            #
            # Imported at module scope, not here: a function-local
            # `import os` makes os a local for the *whole* function, so
            # any earlier branch touching os.environ raises
            # UnboundLocalError -- and an exception inside a D-Bus
            # handler means the method simply never replies, which the
            # client sees as "the portal never answered", 30s later.
            path = os.path.join(
                os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "pipewire-0")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(path)
            fd_list = Gio.UnixFDList.new()
            index = fd_list.append(sock.fileno())
            self._sockets.append(sock)
            invocation.return_value_with_unix_fd_list(
                GLib.Variant("(h)", (index,)), fd_list)
            return
        invocation.return_error_literal(
            Gio.dbus_error_quark(), Gio.DBusError.UNKNOWN_METHOD, method)

    def handle_get(self, _conn, _sender, _path, _iface, prop):
        if prop == "version":
            return GLib.Variant("u", 4)
        if prop == "AvailableSourceTypes":
            return GLib.Variant("u", 1)      # MONITOR
        if prop == "AvailableCursorModes":
            return GLib.Variant("u", 1)      # HIDDEN
        return None


def main():
    node_id = int(sys.argv[1])
    response = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    impl = FakeScreenCastImpl(node_id, response)
    conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    info = Gio.DBusNodeInfo.new_for_xml(INTROSPECTION).interfaces[0]
    conn.register_object(OBJECT_PATH, info, impl.handle_call, impl.handle_get, None)
    Gio.bus_own_name_on_connection(
        conn, BUS_NAME, Gio.BusNameOwnerFlags.NONE, None, None)
    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
