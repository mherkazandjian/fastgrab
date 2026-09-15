# FastGrab

``Fastgrab`` is an opensouce high frame rate screen capture package. A typical
capture frame rate at a resolution of 1080p on a modern machine is ~2600 fps.
There are several other such packages in the wild that are opensource as well, 
but none of them is as fast or provides a simple way of obtaining the captures
image as a numpy array out of the box. The default behavior of ``fastgrab`` is
to provide the user with the image as a numpy array. Beyond that the user is
free to manipulate the image since the pixel data is accessible via a fast and
flexible array, i.e a numpy array.

Typical capture frame rate on a modern machine

resolution    | fps
------------- | -----
360p          | 13900
720p          | 5700
1080p         | 2600
4K            | 340
8K            | 109

Measured with [examples/benchmark.py](https://github.com/mherkazandjian/fastgrab/blob/main/examples/benchmark.py),
800 frames per resolution, on:

 - **CPU**: AMD EPYC 9555 (64-core Zen 5), **8 cores** used for the run
   (``taskset -c 0-7``)
 - **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, driver
   580.178.04 — **not involved in these numbers**. Capture ran against an
   ``Xvfb`` software framebuffer in system RAM, so no pixel goes near the
   GPU. Capturing a real GPU-driven X screen is *slower*, not faster,
   because the framebuffer then lives in VRAM and has to be read back
   across PCIe.
 - **OS**: Ubuntu 22.04.5 LTS, kernel 7.0.0, Python 3.10.12
 - **Display**: ``Xvfb`` at 7680x4320x24 on the same machine, so the
   MIT-SHM fast path is available and large frames get an OpenMP team to
   copy them

A **remote** display cannot use shared memory and falls back to
``XGetImage``, which costs roughly an order of magnitude: on the same
machine that is 221 fps at 1080p rather than 2601. Figures on other
machines will differ — capture is bound by memory bandwidth and, below the
parallel-copy threshold, by single-core clock.

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

## Blurring and redacting regions

``fastgrab.effects`` hides parts of a capture without pulling in Pillow or
OpenCV — it is pure numpy, so it ships in the plain ``pip install fastgrab``.
Pass ``blur=`` to ``capture()``, or call ``blur_regions`` on any BGRA array
you already have:

````python
  from fastgrab import screenshot
  from fastgrab.effects import BlurStyle, blur_regions

  grab = screenshot.Screenshot(blur_style=BlurStyle(method='gaussian', radius=16))
  # regions are (x, y, width, height) in *screen* coordinates
  img = grab.capture(blur=[(100, 100, 400, 200)])

  # or afterwards, on a frame you already have
  blur_regions(img, [(0, 0, 320, 80)], BlurStyle(method='fill', color=(0, 0, 0)))
````

``BlurStyle.method`` picks how a region is obscured:

method     | what it does                          | tune with
---------- | ------------------------------------- | ---------------
``box``    | moving average (default, radius 12)   | ``radius``
``gaussian`` | ``passes`` box blurs, smoother      | ``radius``, ``passes``
``pixelate`` | block means, the mosaic look        | ``block``
``pixelate-random`` | every tile a random colour   | ``block``, ``seed``
``pixelate-random-shuffle`` | the real tile colours, positions permuted | ``block``, ``seed``
``fill``   | a solid ``(B, G, R)`` box, black by default | ``color``
``image``  | a picture stamped over the region | ``image``, ``image_fit``

How much each one destroys, strongest first:

- ``fill``, ``pixelate-random`` and ``image`` — the output does not depend
  on the region's content at all, so nothing of it survives. ``fill`` says
  so plainly; the other two just look friendlier.
- ``pixelate-random-shuffle`` — real tile colours, scrambled positions. The
  region still looks like it belongs, but its colour histogram survives, so
  it leaks roughly "how much of what" was there.
- ``pixelate`` — layout and colour both survive at tile resolution; text can
  be partially recovered by matching candidate renderings to the tile grid.
- ``box`` / ``gaussian`` — weakest, and a low radius is recoverable.

**Use ``fill``, ``pixelate-random`` or ``image`` for passwords, tokens and
anything else that must not leak.** The other three are cosmetic.

``image`` takes a numpy array on the Python API, so the core still needs
nothing but numpy:

````python
  grab.blur_style = BlurStyle(method='image', image=cover)   # (H, W, 3) BGR uint8
````

``image_fit`` (``--blur-image-fit``) decides how a picture is mapped onto a
region of a different shape:

fit | what it does
--- | ---
``crop`` | default; scales to cover the region and trims the overflow evenly
``fit`` | scales so the whole picture is visible, padding the rest with ``color``
``stretch`` | distorts it to the exact region shape
``tile`` | repeats it at its own size

Only ``stretch`` changes the picture's proportions.

The CLI's ``--blur-image PATH`` decodes the file with Pillow, which is
optional (``pip install fastgrab[gui]``) and imported only when you use
that flag.

The random modes are seeded (``seed``, ``--blur-seed``) and therefore
identical on every frame. That is deliberate: re-rolling per frame would
let anyone average a recording back towards the mosaic underneath.

Other things worth knowing:

- Regions are clipped to the frame, and each one is blurred using only the
  pixels inside it, so nothing smears across its edge. Pixels outside the
  regions are left byte-identical, alpha is never touched.
- ``blur=True`` obscures the whole frame; ``blur=False`` on ``capture()``
  overrides a blur set on the constructor for that one call.
- Cost scales with the *region*, not the screen, and is independent of the
  radius. Measured in the dev container: a 400×200 region costs ~1.6 ms
  (``box``) / ~4.3 ms (``gaussian``) / ~0.8 ms (``pixelate``) / ~0.3 ms
  (``fill``), the same on a 1080p or a 4K frame. A *full* 1080p frame is a
  different story — ~64 ms for ``box`` and ~180 ms for ``gaussian`` — so
  full-frame blurring is fine for a screenshot but will hold a recording
  below 30 fps unless you use ``pixelate`` (~23 ms) or ``fill`` (~7 ms).

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
- ``--blur X,Y,W,H`` (repeatable) obscures a screen region in every frame;
  ``--blur-all`` does the whole frame. ``--blur-method`` selects ``box``
  (default), ``gaussian``, ``pixelate``, ``pixelate-random``,
  ``pixelate-random-shuffle``, ``fill`` or ``image``, tuned with
  ``--blur-radius``, ``--blur-block``, ``--blur-seed``,
  ``--blur-color B,G,R``, ``--blur-image PATH`` and
  ``--blur-image-fit {crop,fit,stretch,tile}``. Redaction
  happens on the captured frame, so nothing sensitive reaches ffmpeg —
  and, as above, only ``fill`` truly destroys the pixels:

  ````bash
    fastgrab-record --fullscreen -o demo.mp4 \
        --blur 1200,40,600,120 --blur-method fill --blur-color 0,0,0
  ````

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

   - Debian/Ubuntu: ``sudo apt install build-essential python3-dev libx11-dev libxext-dev``
   - Fedora: ``sudo dnf install gcc python3-devel libX11-devel libXext-devel``

   Runtime needs ``libX11``, ``libXext`` and whichever OpenMP runtime
   the compiler used: ``libgomp1`` for a gcc build (``libgomp`` on
   Fedora), ``libomp`` for a clang one. All are present on any desktop.
   If the toolchain cannot provide OpenMP at build time the extension is
   built without it and says so on stderr — capture still works and
   still uses MIT-SHM, large frames are just copied on one core. That is
   decided when the package is compiled, so installing an OpenMP runtime
   afterwards does not switch it on; reinstall to pick it up.
   ``Python.h: No such file`` means the
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

   Capture uses the **MIT-SHM** X extension when it can: the server
   writes the region straight into a shared memory segment instead of
   pushing every pixel through the X socket, which is worth roughly
   4-10x depending on resolution. It needs client and server on the same
   machine, so a remote ``DISPLAY`` (or a server built without the
   extension) falls back to ``XGetImage`` automatically — one failed
   probe per connection, not per frame. Rows are copied out across an
   OpenMP team once a frame is large enough to pay for it. Two
   environment variables override the defaults, mostly for debugging:
   ``FASTGRAB_NO_XSHM=1`` forces the plain path, and
   ``FASTGRAB_OMP_MIN_BYTES`` sets the frame size in bytes above which
   the copy is parallelised (``0`` keeps it serial always).

   **After a fork, the copy goes back to being serial.** GNU libgomp is
   not fork-safe: a process that has run a parallel region keeps its
   thread pool, ``fork()`` copies the bookkeeping but not the threads,
   and the child's next parallel region waits forever for workers that do
   not exist. So a child that inherited fastgrab — a ``multiprocessing``
   worker under the ``fork`` start method, say — copies serially. It
   keeps the shared-memory path, which is the larger half of the win, and
   gives up roughly a quarter of the combined speed at 1080p.

   Processes that get a clean runtime are unaffected: the ``spawn`` start
   method, or a ``forkserver`` whose server process never captured.

   Importing fastgrab inside the worker instead of preloading it helps
   only when nothing in the parent had already started an OpenMP pool.
   The guard stamps its pid when the extension is imported, so a worker
   that imports *after* the fork looks like an ordinary first import and
   re-arms the parallel copy — which still deadlocks if some other
   library warmed libgomp before the fork. ``spawn`` is the answer that
   does not depend on knowing what else is in the process.

   If you know your children fork from a runtime that never started an
   OpenMP pool, ``FASTGRAB_UNSAFE_OMP_AFTER_FORK=1`` turns the guard off
   and keeps the parallel copy everywhere. It is named that way on
   purpose: fastgrab cannot verify the precondition, and getting it wrong
   is a hang rather than an error.
 - **Linux/Wayland (wlroots)**: a wlroots-based compositor (Sway,
   Hyprland, river, niri, cage) for the no-prompt path; the ``[wayland]``
   extra (``pip install fastgrab[wayland]``) pulls in ``pywayland``.
 - **Linux/Wayland (GNOME, KDE)**: those desktops do not implement
   ``wlr-screencopy-v1``, so capture goes through ``xdg-desktop-portal``
   and PipeWire instead — the same mechanism screen-sharing in a browser
   uses. ``pip install fastgrab[portal]`` brings PyGObject; GStreamer and
   its introspection data are system packages pip cannot install:

   ````bash
   sudo apt install gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
                    gstreamer1.0-plugins-base gstreamer1.0-pipewire
   ````

   That is the Debian/Ubuntu set the ``test-portal`` image is built from,
   so it is the one that is exercised. Other distributions ship the same
   pieces under their own names (Fedora: ``gstreamer1-plugins-base``,
   ``gstreamer1-plugin-pipewire``, ``python3-gobject``) — untested here.
   If pip has to *build* PyGObject rather than reuse a distro
   ``python3-gi``, it also needs ``libgirepository1.0-dev``,
   ``libcairo2-dev``, ``pkg-config`` and a C toolchain. The extra pins
   PyGObject below 3.51 for that reason: from 3.51 it wants
   ``girepository-2.0``, which ``libgirepository1.0-dev`` does not
   provide and which only reaches Debian in trixie.

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
