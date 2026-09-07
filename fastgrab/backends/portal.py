"""xdg-desktop-portal + PipeWire fallback backend.

Status: **stub, deliberately.**

This file used to justify itself by saying the flow "cannot be
exercised inside docker compose CI (the consent dialog on GNOME/KDE is
a fundamental Wayland-security feature, not something we can mock away)".
A spike in 2026-09 showed that reason was too strong, and found a
narrower one that does hold. Both halves are recorded here so nobody
has to run the experiment again.

What the spike disproved
------------------------

The consent dialog is a GNOME/KDE *portal backend* behaviour, not part
of the ScreenCast API. ``xdg-desktop-portal-wlr`` picks an output with
no UI at all when told to, via
``$XDG_CONFIG_HOME/xdg-desktop-portal-wlr/config``::

    [screencast]
    chooser_type=none

which upstream implements as a straight list pick — no process
spawned, nothing drawn (``src/screencast/wlr_screencast.c``,
``xdpw_wlr_output_chooser``)::

    case XDPW_CHOOSER_NONE:
        if (ctx->state->config->screencast_conf.output_name) {
            return xdpw_wlr_output_find_by_name(&ctx->output_list, ...);
        } else {
            return xdpw_wlr_output_first(&ctx->output_list);
        }

Driven that way inside the existing ``test-wayland`` cage session
(Debian 13, xdg-desktop-portal 1.20.3, xdg-desktop-portal-wlr 0.7.1,
pipewire 1.4.2, wireplumber 0.5.8, headless cage), the whole flow ran
unattended::

    CreateSession      -> response 0, session handle
    SelectSources      -> response 0   (types=1 monitor, cursor_mode=1)
    Start              -> response 0, streams [(42, {size: (1280, 720)})]
    OpenPipeWireRemote -> unix fd

and one frame pulled off node 42 was byte-exact against what
``tests/wayland_painter.py`` had just painted: 1280*720*4 = 3686400
bytes, every pixel BGRA ``(30, 20, 10, 255)`` for ``#0A141E``. So the
flow is implementable, and it *is* drivable with no human in the loop.
The old objection, as written, was wrong.

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

The declared extra is also wrong
--------------------------------

Two things the spike found about ``[wayland-portal]`` itself, both of
which a future implementer has to settle before writing any flow code:

* ``pipewire-python`` cannot do the job. It is a subprocess wrapper
  around the ``pw-cat`` / ``pw-play`` / ``pw-record`` CLIs; its entire
  public API is ``playback`` / ``record`` / ``get_list_targets``, it is
  audio-only, and it can neither target a node id, nor accept the
  portal's fd, nor hand back a raw video buffer. It also installs as
  ``pipewire_python``, so the ``import pipewire`` guard this file used
  to run raised ImportError even when the extra *was* installed. The
  spike's frame came from ``gst-launch-1.0 pipewiresrc fd=N
  path=<node_id>``; a real backend would need PyGObject + GStreamer, or
  a hand-written ctypes/cffi binding to ``libpipewire`` including SPA
  POD marshalling. Neither is small, and both are a heavy tax on a
  project whose default install is meant to be numpy and a C extension.

* ``dbus-next`` 0.2.3 cannot introspect the portal at all::

      dbus_next.errors.InvalidMemberNameError:
          invalid member name: power-saver-enabled

  xdg-desktop-portal >= 1.18 exposes a hyphenated property name, and
  dbus-next rejects it (correctly, per the D-Bus spec) while parsing
  the whole introspection document — so a single unrelated interface
  takes down ``bus.introspect()``. It is workable: hand-write the
  introspection XML for ``org.freedesktop.portal.ScreenCast`` and never
  call ``introspect()``. But dbus-next has been unmaintained since 2021
  and this is the first thing it gets wrong.

What would have to change
-------------------------

Any one of these reopens the question:

* xdpw grows a SHM-only path for screencopy v3 (upstream #289); or the
  cage/wlroots in the test image starts advertising
  ``ext-image-copy-capture-v1``, which xdpw master already prefers over
  zwlr-screencopy and which may not carry the dmabuf requirement. The
  cage in Debian 13 does not advertise it.
* CI gains a DRM device — a ``vkms`` module on the runner plus
  ``--device /dev/dri``, or a self-hosted runner with a GPU.
* fastgrab accepts a GStreamer/PyGObject dependency behind the extra.

Until one of those lands, the conclusion is the one this file already
reached, for a better reason: a portal backend written today would be
untestable in this project's CI, so it is not written.

**This says nothing about GNOME/KDE.** The spike was driven entirely
against xdg-desktop-portal-wlr. GNOME and KDE portal backends do show a
consent dialog on ``Start``, and no config key turns that off — that
part of the original docstring stands untouched, and a wlroots-only
test would never have covered it.

Reproducing the spike
---------------------

Roughly, on a host with a Mesa-supported GPU (all of it inside a
container built ``FROM fastgrab-dev`` with ``dbus pipewire wireplumber
xdg-desktop-portal xdg-desktop-portal-wlr gstreamer1.0-pipewire
gstreamer1.0-tools libgl1-mesa-dri libegl-mesa0 libgbm1`` added)::

    export XDG_RUNTIME_DIR=/tmp/xdg-runtime XDG_CURRENT_DESKTOP=wlroots
    export WLR_BACKENDS=headless WLR_RENDER_DRM_DEVICE=/dev/dri/renderD129
    # config as above; then, under dbus-run-session:
    cage -s -- python tests/wayland_painter.py &
    pipewire & wireplumber &
    /usr/libexec/xdg-desktop-portal-wlr -l TRACE &
    /usr/libexec/xdg-desktop-portal -v &
    # then CreateSession/SelectSources/Start/OpenPipeWireRemote over
    # the session bus, and
    #   gst-launch-1.0 pipewiresrc fd=$FD path=$NODE num-buffers=1 \
    #     ! videoconvert ! video/x-raw,format=BGRA ! filesink location=f.bgra

Drop ``--device /dev/dri`` and it fails at ``Start`` with response code
2 and the xdpw log line quoted above. That is the whole finding.

When it is implemented, the flow is:

1. Connect to the D-Bus session bus.
2. Proxy ``org.freedesktop.portal.Desktop`` at
   ``/org/freedesktop/portal/desktop`` with interface
   ``org.freedesktop.portal.ScreenCast`` — from hand-written
   introspection XML, not ``bus.introspect()``, per the dbus-next note
   above.
3. ``CreateSession(handle_token, session_handle_token)`` → wait for the
   ``Response`` signal on the returned ``org.freedesktop.portal.Request``
   path, capture ``session_handle``.
4. ``SelectSources(session_handle, types=1, multiple=False, cursor_mode=...)``.
5. ``Start(session_handle, parent_window="", options={})`` —
   **this is when GNOME/KDE pop the consent dialog**, and where the
   wlroots backend fails without dmabuf. Response yields
   ``streams: [(node_id, props), ...]``; ``props["size"]`` is the
   device-pixel size of the source. Cache ``node_id``.
6. ``OpenPipeWireRemote(session_handle, options)`` returns a unix fd for
   a pre-authenticated PipeWire connection. It is single-use: each
   consumer connection needs a fresh call.
7. Per-capture: connect a PipeWire stream to ``node_id`` over that fd,
   negotiate a format (do **not** assume BGRA — check what the node
   actually offers and convert), pull one buffer, copy into the
   caller's numpy ndarray, and keep the session alive across captures
   rather than re-running the flow per frame.

Probe the portal's ``interface_version`` after binding and degrade
options it does not advertise. The spike saw version 5,
``AvailableSourceTypes=1`` (monitor only) and
``AvailableCursorModes=3`` from xdpw 0.7.1; Fedora's
xdg-desktop-portal-gnome and Ubuntu's -gtk differ on ``cursor_mode`` /
``persist_mode``.
"""
from .base import BaseBackend


_NOT_IMPLEMENTED_MSG = (
    "PortalBackend is not implemented. The ScreenCast flow itself works "
    "unattended against xdg-desktop-portal-wlr, but it cannot be tested "
    "in this project's CI (xdg-desktop-portal-wlr needs a DMA-BUF capable "
    "GPU, which the runners do not have; see the module docstring), and "
    "the wayland-portal extra does not currently pull in a library that "
    "can read raw video buffers from PipeWire. On GNOME/KDE Wayland "
    "sessions, log in to an X11 session for now, or use a wlroots "
    "compositor (Sway, Hyprland) so the wlr backend works without a "
    "consent prompt."
)


class PortalBackend(BaseBackend):
    def __init__(self):
        # Only dbus-next is worth checking: it is the one declared
        # dependency of the extra that a real implementation would still
        # use. pipewire-python is *not* checked because it cannot serve
        # this backend at all (audio-only pw-cat wrapper) — and the guard
        # this file used to run, `import pipewire`, could never have
        # succeeded anyway, since the distribution installs the module as
        # `pipewire_python`. See the module docstring.
        try:
            import dbus_next  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "wayland-portal extra is not installed: "
                "pip install fastgrab[wayland-portal]"
            ) from exc

        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def resolution(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
