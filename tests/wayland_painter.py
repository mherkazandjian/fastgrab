"""Test-only Wayland kiosk client for the wlr integration suite.

Runs as cage's child app. Cage gives this client the entire output. The
painter listens on the UNIX socket at ``$XDG_RUNTIME_DIR/fastgrab-painter.sock``
for newline-terminated commands of the form::

    paint #RRGGBB

and redraws its fullscreen surface in that solid color (premultiplied
ARGB8888 SHM buffer). The surface is set to fullscreen and the painter
responds to ``xdg_toplevel.configure`` events to keep its buffer the
same size as the output the compositor gives it — otherwise solid-color
asserts in the wlr integration tests would see uncovered border pixels.

This file intentionally has no production dependency — it lives under
``tests/`` and is only invoked by the test compose service.
"""
from __future__ import annotations

import mmap
import os
import select
import socket
import struct
import sys

from pywayland.client import Display
from pywayland.protocol.wayland import WlCompositor, WlShm
from pywayland.protocol.xdg_shell import XdgWmBase


SOCKET_NAME = "fastgrab-painter.sock"
DEFAULT_W = 1280
DEFAULT_H = 720


class Painter:
    def __init__(self):
        self.compositor = None
        self.shm = None
        self.xdg_wm_base = None
        self.surface = None
        self.xdg_surface = None
        self.xdg_toplevel = None
        self.wl_buffer = None
        self.mm = None
        self.fd = None
        self.size = None  # (w, h, stride) currently allocated
        self.target_w = DEFAULT_W
        self.target_h = DEFAULT_H
        self.color_bgra = (0, 0, 0, 0xFF)
        self.configured = False

        self.display = Display()
        self.display.connect()

        registry = self.display.get_registry()
        registry.dispatcher["global"] = self._on_global
        self.display.roundtrip()
        self.display.roundtrip()

        if self.compositor is None or self.shm is None or self.xdg_wm_base is None:
            raise RuntimeError(
                "compositor missing required globals: "
                "compositor={} shm={} xdg_wm_base={}".format(
                    self.compositor, self.shm, self.xdg_wm_base
                )
            )

        self.xdg_wm_base.dispatcher["ping"] = lambda b, serial: b.pong(serial)

        self.surface = self.compositor.create_surface()
        self.xdg_surface = self.xdg_wm_base.get_xdg_surface(self.surface)
        self.xdg_surface.dispatcher["configure"] = self._on_xdg_surface_configure
        self.xdg_toplevel = self.xdg_surface.get_toplevel()
        self.xdg_toplevel.dispatcher["configure"] = self._on_toplevel_configure
        self.xdg_toplevel.set_title("fastgrab-painter")
        # Ask the compositor to size us to the entire output; cage honours this.
        self.xdg_toplevel.set_fullscreen(None)
        self.surface.commit()

        # Wait for the first xdg_surface configure → ack + draw.
        while not self.configured:
            self.display.dispatch(block=True)

    def _on_global(self, registry, name, interface, version):
        if interface == "wl_compositor":
            self.compositor = registry.bind(name, WlCompositor, min(version, 4))
        elif interface == "wl_shm":
            self.shm = registry.bind(name, WlShm, min(version, 1))
        elif interface == "xdg_wm_base":
            self.xdg_wm_base = registry.bind(name, XdgWmBase, min(version, 3))

    def _on_toplevel_configure(self, toplevel, width, height, states):
        # 0 means "you decide"; anything positive is the compositor's pick.
        if width > 0:
            self.target_w = width
        if height > 0:
            self.target_h = height

    def _on_xdg_surface_configure(self, xdg_surface, serial):
        xdg_surface.ack_configure(serial)
        self._ensure_buffer(self.target_w, self.target_h)
        self._redraw()
        self.configured = True

    def _ensure_buffer(self, w, h):
        stride = w * 4
        size = stride * h
        if self.size == (w, h, stride):
            return
        # Tear down the previous buffer if we had one.
        if self.wl_buffer is not None:
            try:
                self.wl_buffer.destroy()
            except Exception:
                pass
        if self.mm is not None:
            try:
                self.mm.close()
            except Exception:
                pass
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass

        self.fd = os.memfd_create("fastgrab-painter", 0)
        os.ftruncate(self.fd, size)
        self.mm = mmap.mmap(
            self.fd, size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            flags=mmap.MAP_SHARED,
        )
        pool = self.shm.create_pool(self.fd, size)
        self.wl_buffer = pool.create_buffer(0, w, h, stride, 0)  # ARGB8888
        pool.destroy()
        self.size = (w, h, stride)

    def _fill(self, b, g, r, a, w, h):
        pixel = struct.pack("<BBBB", b, g, r, a)
        self.mm.seek(0)
        self.mm.write(pixel * (w * h))

    def _redraw(self):
        b, g, r, a = self.color_bgra
        w, h, _ = self.size
        self._fill(b, g, r, a, w, h)
        self.surface.attach(self.wl_buffer, 0, 0)
        self.surface.damage(0, 0, w, h)
        self.surface.commit()
        self.display.flush()

    def paint(self, hex_color):
        if not hex_color.startswith("#") or len(hex_color) != 7:
            raise ValueError("expected #RRGGBB, got {!r}".format(hex_color))
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        self.color_bgra = (b, g, r, 0xFF)
        self._redraw()


def main():
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        print("XDG_RUNTIME_DIR is not set", file=sys.stderr)
        sys.exit(2)

    sock_path = os.path.join(runtime, SOCKET_NAME)
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(4)
    srv.setblocking(False)

    painter = Painter()
    print("painter ready", flush=True)

    wl_fd = painter.display.get_fd()
    clients = []  # list of (sock, recv_buffer)

    while True:
        painter.display.flush()
        rlist = [srv, wl_fd] + [c for c, _ in clients]
        r, _, _ = select.select(rlist, [], [], None)

        if wl_fd in r:
            painter.display.dispatch(block=False)

        if srv in r:
            try:
                cs, _ = srv.accept()
                cs.setblocking(False)
                clients.append((cs, b""))
            except BlockingIOError:
                pass

        new_clients = []
        for cs, buf in clients:
            if cs in r:
                try:
                    chunk = cs.recv(4096)
                except (BlockingIOError, ConnectionResetError):
                    chunk = b""
                if not chunk:
                    cs.close()
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip().decode("utf-8", "replace")
                    try:
                        if line.startswith("paint "):
                            painter.paint(line[6:].strip())
                            cs.sendall(b"ok\n")
                        elif line == "ping":
                            cs.sendall(b"pong\n")
                        else:
                            cs.sendall(b"err: unknown command\n")
                    except Exception as e:
                        cs.sendall(("err: " + str(e) + "\n").encode())
                new_clients.append((cs, buf))
            else:
                new_clients.append((cs, buf))
        clients = new_clients


if __name__ == "__main__":
    main()
