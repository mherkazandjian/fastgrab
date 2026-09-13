# FastGrab

``Fastgrab`` is an opensouce high frame rate screen capture package. A typical
capture frame rate at a resolution of 1080p on a modern machine is ~60 fps.
There are several other such packages in the wild that are opensource as well, 
but none of them is as fast or provides a simple way of obtaining the captures
image as a numpy array out of the box. The default behavior of ``fastgrab`` is
to provide the user with the image as a numpy array. Beyond that the user is
free to manipulate the image since the pixel data is accessible via a fast and
flexible array, i.e a numpy array.

Typical capture frame rate on a modern machine

resolution    | fps
------------- | -----
360p          | > 800
720p          | 260
1080p         | 200
4K            | 20

# Usage example

````python
  from fastgrab import screenshot
  # take a full screen screenshot
  img = screenshot.Screenshot().capture()
  # >> img is a numpy ndarray of shape (height, width, 4) in BGRA byte order
  # >> do whatever you want with it
  # (optional)
  # e.g it can be displayed with matplotlib (install matplotlib first)
  from matplotlib import pyplot as plt
  # matplotlib expects RGB, so reverse the BGR channels and drop alpha
  plt.imshow(img[:, :, 2::-1], interpolation='none')
  plt.show()
````

Reuse one ``Screenshot`` when you are capturing repeatedly — it keeps the
buffer and the connection to the display, so a loop over ``capture()`` is
much cheaper than a loop over ``Screenshot().capture()``. If you do build
them per frame, each one holds an OS-level frame buffer until it is
collected; ``close()``, or the context manager, releases it at a moment
you choose:

````python
  from fastgrab import screenshot

  with screenshot.Screenshot() as grab:
      for _ in range(100):
          img = grab.capture((0, 0, 640, 480))
````

``capture()`` refuses to run on a closed instance rather than returning a
stale frame.

## Screen recording (draft, Linux/X11)

The opt-in ``fastgrab.recording`` module pipes frames into ``ffmpeg``
(which must be on ``PATH``). Pick a capture target — ``--fullscreen`` or
``--region X,Y,W,H`` for scripted use, or ``--gui`` to select interactively
(``--gui`` can be combined with either to skip the drag selector and go
straight to the settings dialog) — and an output whose extension selects
the codec (``.mp4``, ``.webm`` or ``.gif``):

````bash
  fastgrab-record --fullscreen --duration 10 -o demo.mp4
  fastgrab-record --region 100,100,1280,720 --fps 60 -o clip.webm   # Ctrl-C to stop
  fastgrab-record --fullscreen --countdown 3 --title "My demo" --overlay-text "v1.2" -o demo.mp4
````

- ``--fps N`` sets the target rate (default 30). If capture runs slower
  than that, the last frame is repeated so the clip's length still matches
  wall-clock time; the summary line reports the real capture rate and how
  many frames were duplicated.
- ``--duration S`` stops after S seconds; without it, recording runs until
  Ctrl-C and the file is finalised cleanly. Ctrl-C during the countdown
  cancels instead: nothing has been encoded, so no file is written and the
  command says so rather than naming one. A Ctrl-C before recording starts
  at all — while the region selector is open, say — exits 130 without a
  traceback.
- ``--countdown S`` waits before the first frame — time to move the
  terminal out of shot.
- ``--title TEXT`` shows top-centre for the first 3 seconds;
  ``--overlay-text TEXT`` is a watermark in the top-right for the whole clip.
  Both are drawn literally: an apostrophe, a colon, a backslash or a percent
  sign all come out as typed. That does mean ffmpeg's own ``%{...}``
  expansions are *not* interpreted — a title reading ``%{pts}`` renders
  those characters rather than a timestamp.

Optional pointer overlays and subtitles:

````bash
  fastgrab-record --fullscreen -o demo.mp4 \
      --show-clicks --click-style concentric --click-color 255,200,0 \
      --show-cursor \
      --subtitle "0.5-3.0:Hello world" --subtitle "4.0-6.5:Second line" \
      --subtitle-font /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
      --subtitle-fontsize 32 --subtitle-color yellow \
      --subtitle-box-color black@0.6 --subtitle-position bottom
````

- ``--show-clicks`` animates every detected mouse click; ``--click-style``
  picks the pattern (``ring``, ``concentric``, ``circle``, ``crosshair``),
  ``--click-color B,G,R``/``--click-lifetime`` tune it.
- ``--show-cursor`` stamps an emulated arrow pointer at the mouse position —
  the X11 capture path never includes the real cursor sprite.
- ``--subtitle START-END:TEXT`` (repeatable) renders timed subtitles;
  colours accept any ffmpeg colour string (``white``, ``0xRRGGBB``,
  ``red@0.8``). The same options exist on the Python API via
  ``ClickStyle``, ``Subtitle`` and ``SubtitleStyle``.

Click/cursor tracking and ``--gui`` need the ``[gui]`` extra
(``pip install fastgrab[gui]``, which pulls in ``python-xlib`` for pointer
polling and ``Pillow`` for the selector's preview); subtitles need a font
file (``$FASTGRAB_FONT`` or the bundled DejaVu search paths).

### ASS subtitles

By default every subtitle becomes an ffmpeg ``drawtext`` filter.
Two timing notes for the ASS backend: a cue's end time is exclusive, so
it stops one frame earlier than the drawtext chain does, and ASS
timestamps are centiseconds, so a cue shorter than 10 ms is refused
rather than silently written as one that can never appear.

``--subtitle-backend ass`` instead generates an Advanced SubStation Alpha
script from the same ``--subtitle`` lines and burns it in with libass
(needs an ffmpeg built ``--enable-libass``, which the Debian/Ubuntu one
is). ``--subtitle-sidecar`` keeps that script as an editable file —
named after the video when no path is given, so ``demo.mp4`` gets
``demo.ass``, which mpv and VLC pick up on their own as a track the
viewer can switch off. The two flags are independent: a sidecar can
accompany drawtext burn-in just as well.

````bash
  fastgrab-record --fullscreen -o demo.mp4 \
      --subtitle "0.5-3.0:Hello world" --subtitle "4.0-6.5:Second line" \
      --subtitle-backend ass --subtitle-sidecar \
      --subtitle-font-name "DejaVu Sans" --subtitle-color yellow
````

From Python the same options are ``Recorder(..., subtitle_backend="ass",
subtitle_sidecar="demo.ass")``, and ``build_ass_document(subtitles,
style)`` returns the script as a string when the file is all you want.

Both backends read the same ``SubtitleStyle``, which the ASS one maps
onto a ``[V4+ Styles]`` row: font size, colours (converted to
``&HAABBGGRR``, where the alpha byte is inverted), ``border`` as the
opaque box's padding, and ``position`` as the alignment plus vertical
margin. Three differences are worth knowing about:

- libass resolves fonts by *family name*, not by path, so the family is
  guessed from ``--subtitle-font``'s filename — pass
  ``--subtitle-font-name`` when that guess is wrong. Unlike drawtext,
  the ASS backend still renders when no font file is found at all.
- the colour converter understands ``0xRRGGBB[AA]``, ``#RRGGBB[AA]`` and
  ~25 common colour names, not ffmpeg's full ~150-name table; anything
  else is refused before the encode starts rather than silently
  mis-coloured.
- long lines wrap inside the margins instead of running off the edge the
  way drawtext does.

Interactive use: ``fastgrab-record --gui`` opens a drag-to-select region
picker followed by a small settings dialog (needs ``tkinter``), and
``fastgrab-record --print-xbindkeys`` prints a snippet for binding that to
a hotkey such as ``Print``.

From Python:

````python
  from fastgrab.recording import Recorder
  stats = Recorder("demo.mp4", bbox=(0, 0, 1280, 720), fps=30).record(duration=5)
  # stats: frames (captured), written_frames (incl. duplicates),
  #        elapsed_seconds, achieved_fps, output
````

## Getting Started

``Fastgrab`` was initially developed in 2016 as part of an aimbot (for quake
live). It supports **Linux** (X11 via a libX11 C extension; Wayland via
``wlr-screencopy-v1`` behind the ``[wayland]`` extra, or
``xdg-desktop-portal`` + PipeWire behind the ``[portal]`` extra for GNOME
and KDE), **Windows 10/11** (Win32 ``BitBlt`` via ``ctypes``), and **macOS**
(CoreGraphics ``CGDisplayCreateImage`` via ``ctypes``). The plain
``pip install fastgrab`` works on all three platforms; only Wayland needs
an opt-in extra.

## Comparison with other packages

The following comparison has been done a Intel i7-6700HQ with 16 GB ram  at a
1080p resolution. ``fastgrab`` is designed to be fast and
does not provide any features beyond capturing the screen, unlike the other
packages mentioned in  the comparison below that do many great things. 

package       | fps
------------- | -----
fastgrab      | 200
python-mss    | 180
autopy        | 34
pyautogui     | 8
pyscreenshot  | 4

to benchmark ``fastgrab`` run the script [examples/benchmark.py](https://github.com/mherkazandjian/fastgrab/blob/main/examples/benchmark.py)

### Prerequisites

Common to all platforms:

 - ``python >= 3.10`` (python 2 is not supported)
 - ``Numpy >= 1.26`` (auto-installed by pip)

Per-platform extras:

 - **Linux/X11**: the C extension is compiled on install, so you need a C
   toolchain plus the Python and X11 headers:

   - Debian/Ubuntu: ``sudo apt install build-essential python3-dev libx11-dev``
   - Fedora: ``sudo dnf install gcc python3-devel libX11-devel``

   Runtime needs only ``libX11`` and ``libgomp1`` (``libgomp`` on Fedora),
   which are present on any desktop. ``Python.h: No such file`` means the
   Python headers are missing (``python3-dev``); ``stdio.h: No such file``
   means the toolchain headers are (``build-essential``). Tested on
   Debian-based Python images, Ubuntu 24.04 and Fedora 44, Python 3.10–3.14.

   The X server must be running at **depth 24** (the default). A depth-30
   screen — 10-bit colour, which some drivers offer for HDR-ish setups —
   packs 10:10:10 RGB into the same 32 bits per pixel, so its channels do
   not line up on byte boundaries and cannot be handed back as BGRA.
   fastgrab refuses that layout with a message naming the masks it found
   rather than returning wrong colours. ``xdpyinfo | grep "depth of root"``
   says which you have.
 - **Linux/Wayland (wlroots)**: a wlroots-based compositor (Sway,
   Hyprland, river, niri, cage) for the no-prompt path; the ``[wayland]``
   extra (``pip install fastgrab[wayland]``) pulls in ``pywayland``.
 - **Linux/Wayland (GNOME, KDE)**: those desktops do not implement
   ``wlr-screencopy-v1``, so capture goes through ``xdg-desktop-portal``
   and PipeWire instead — the same mechanism screen-sharing in a browser
   uses. ``pip install fastgrab[portal]`` brings PyGObject; GStreamer and
   its introspection data are system packages pip cannot install:

   - Debian/Ubuntu: ``sudo apt install python3-gi gir1.2-gst-plugins-base-1.0 gstreamer1.0-pipewire``
   - Fedora: ``sudo dnf install python3-gobject gstreamer1-plugins-base gstreamer1-plugin-pipewire``

   **This path asks your permission.** The desktop shows its own
   screen-share chooser the first time a ``Screenshot`` captures, and the
   approval belongs to that object — so reuse one and you are asked once,
   build a new one per frame and you are asked per frame:

   ````python
   grab = screenshot.Screenshot()      # nothing is asked here
   while True:
       img = grab.capture()            # asked once, on the first capture
   ````

   Whether the desktop may *remember* the approval is yours to choose,
   with ``$FASTGRAB_PORTAL_PERSIST`` or the ``persist`` argument:

   - ``none`` — ask every time a session starts
   - ``transient`` — remember until you log out (**the default**)
   - ``persistent`` — remember across reboots, via a token the desktop
     stores

   ````bash
   FASTGRAB_PORTAL_PERSIST=persistent python yourscript.py
   ````

   ````python
   screenshot.Screenshot(backend="portal", persist="none")
   ````

   ``persistent`` is the least clicking and the most trust: the desktop
   keeps a token that lets fastgrab re-open the same share without asking
   again. ``none`` is the opposite. The default sits between the two and
   never survives a logout.
 - **Windows 10/11**: nothing beyond Python + numpy. Capture goes through
   GDI ``BitBlt`` via ``ctypes``.
 - **macOS**: nothing beyond Python + numpy. macOS 10.15+ requires
   *Screen Recording* permission for the running app (System Settings →
   Privacy & Security). Captures are in **device pixels**, so a Retina
   display reports (and returns) the full backing store — a 1800x1169
   desktop captures as 3600x2338. ``bbox`` rectangles are in device
   pixels too.

note that ``fastgrab`` could work with lower versions but I have not tested it
(and probaby will not). 

### Installing

``Fastgrab`` can be installed in several ways:

```bash
pip install fastgrab
```

```bash
pip install git+https://github.com/mherkazandjian/fastgrab.git
```

```bash
git clone https://github.com/mherkazandjian/fastgrab.git
cd fastgrab
pip install .
```

## Running the tests

The canonical way to run the test suite is through the project's docker
compose setup, which bundles ``Xvfb``, ``libX11`` and the build toolchain
so tests are reproducible regardless of the host:

````bash
make test docker=1
````

which is equivalent to

````bash
docker compose run --rm test
````

Two further suites cover the Wayland backends, and are separate because
each needs a display stack of its own:

````bash
docker compose run --rm test-wayland   # wlr-screencopy, under headless cage
docker compose run --rm test-portal    # xdg-desktop-portal + PipeWire
````

Neither needs a GPU. The portal suite runs the real portal frontend with
a fake desktop chooser behind it, so the protocol is exercised for real
without anyone clicking "Share".

If you have ``pytest``, ``numpy``, ``python-xlib`` and an X server (or
``xvfb-run``) available on the host, the suite also runs directly:

````bash
make test
````

The ``Makefile`` exposes ``build``, ``install``, ``dev`` (a virtual
desktop on ``localhost:5901`` over VNC), ``benchmark``, ``lock`` and
``clean`` targets — run ``make`` with no arguments for the help listing.

## Contributing

Submit a pull request or create an [issue](https://github.com/mherkazandjian/fastgrab/issues/new)
if you find any bugs.

Any help/pull requests are welcome. The default wheel must stay
dependency-light; additional backends or features ship as opt-in pip
extras (``pip install fastgrab[<extra>]``) when they have non-trivial
runtime deps. Open follow-ups include:

   - macOS ``ScreenCaptureKit`` backend for Apple-Silicon-era systems
   - Multi-monitor capture across all backends

## Authors

* **Mher Kazandjian** - [Github](https://github.com/mherkazandjian)

## License

This project is licensed under GPLv3

## Acknowledgments

* pyscreenshot
* autopy
* pyautogui
* reame template taken from: [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* https://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux/16141058#16141058
* python-mss
