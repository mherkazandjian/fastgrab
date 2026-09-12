"""
Module that implements the object for taking screenshots
"""
import numbers

import numpy

from fastgrab.backends import _resolve_backend


def _normalise_blur(blur):
    """Validate blur regions and materialise them into a tuple.

    ``blur`` is ``None`` / ``True`` / ``False`` or an iterable of
    ``(x, y, width, height)``. The rectangle checks live in
    :func:`fastgrab.effects._normalise_regions` so that the low-level
    entry point enforces exactly the same rules; importing it here is
    still lazy, because only a caller who passed actual regions reaches
    that line.
    """
    if blur is None or blur is True or blur is False:
        return blur
    try:
        regions = tuple(blur)
    except TypeError:
        raise TypeError(
            "blur must be None, True, False, or an iterable of "
            "(x, y, width, height); got {!r}".format(type(blur).__name__)
        )
    if not regions:
        if hasattr(blur, "__next__"):
            # An empty *iterator* is almost always one that was already
            # consumed by an earlier call — silently capturing in the
            # clear is exactly the failure this feature must not have.
            # An intentional no-op is spelled [] or False.
            raise ValueError(
                "blur regions iterable was empty or already consumed; "
                "pass [] or False to capture unmodified, and store a "
                "list rather than a generator to reuse regions"
            )
        # A genuinely empty list is documented as "capture unmodified",
        # so it must not drag in fastgrab.effects. Materialising first
        # (the iterable is consumed either way) is what lets this be
        # decided before the import.
        return ()
    from fastgrab.effects import _normalise_regions
    return _normalise_regions(regions)


class Screenshot(object):
    """
    Main object that captures screenshots and provides other utilities
    """
    def __init__(self, backend=None, blur=None, blur_style=None):
        """
        Constructor

        :param backend: optional explicit backend name — one of
            ``'x11'``, ``'wlr'``, ``'portal'``. When ``None`` (default)
            the backend is auto-detected from the environment:
            Wayland sessions try wlr → portal; X11 sessions use x11.
        :param blur: regions to obscure in every capture — a list of
            screen-absolute ``(x, y, width, height)`` rectangles, or
            ``True`` for the whole frame. ``None`` (default) captures
            unmodified frames. See :meth:`capture`.
        :param blur_style: a :class:`fastgrab.effects.BlurStyle`
            selecting the method (``box``, ``gaussian``, ``pixelate`` or
            a solid ``fill``) and its parameters; ``None`` uses the
            defaults.
        """
        self._backend = _resolve_backend(backend)
        """The capture backend (BaseBackend subclass instance)"""

        self._screensize = None
        """tuple, (width, height), backing variable for self.screensize"""

        self._img = None
        """The buffer where the captured image is stored"""

        self._closed = False
        """Set by :meth:`close`; capture refuses to run afterwards"""

        self.blur = blur              # both normalised/validated by the
        self.blur_style = blur_style  # property setters defined below

        self._blur_scratch = {}
        """Work buffers reused across captures by fastgrab.effects, so a
        capture loop blurring a fixed region stops reallocating them"""

    @property
    def blur(self):
        """Regions obscured in every capture, or ``None`` / ``True``.

        A property rather than a plain attribute so that assigning to it
        after construction goes through the same validation the
        constructor uses — otherwise ``grab.blur = (r for r in rects)``
        would redact one frame and then silently stop.
        """
        return self._blur

    @blur.setter
    def blur(self, value):
        self._blur = _normalise_blur(value)

    @property
    def blur_style(self):
        """The :class:`fastgrab.effects.BlurStyle` used for :attr:`blur`.

        Validated on assignment for the same reason as :attr:`blur`, and
        so that a bad style is refused up front rather than at the end of
        the next ``capture()`` — by which point the backend has already
        overwritten the shared buffer that a caller may still be holding
        as a redacted frame.
        """
        return self._blur_style

    @blur_style.setter
    def blur_style(self, value):
        if value is not None:
            from fastgrab.effects import BlurStyle
            if not isinstance(value, BlurStyle):
                raise TypeError(
                    "blur_style must be a BlurStyle or None, got "
                    "{!r}".format(type(value).__name__)
                )
        self._blur_style = value

    @property
    def screensize(self) -> tuple:
        """
        return the screensize/resolution
        """
        if self._screensize is None:
            self._screensize = self._backend.resolution()
            return self._screensize
        else:
            return self._screensize

    def refresh(self):
        """
        Forget cached display geometry so the next capture re-reads it

        :attr:`screensize` is cached after its first read — capture
        loops ask for it on every frame, and re-querying the display
        each time would put a round trip in the hot path. The cost is
        that a resolution change, a monitor being plugged in, or a
        different display becoming the main one goes unnoticed: full
        screen captures keep covering the old region and boxes in the
        newly available area are rejected as out of bounds.

        Call this when the display setup changes. It asks the backend to
        re-resolve whatever it latched onto at construction, then drops
        the cached size and the capture buffer.

        What a backend can actually notice varies: x11 and windows query
        the display on every call, so nothing is latched; macOS re-reads
        which display is the main one; wlr re-reads output geometry and
        re-selects its output, but does not track outputs disappearing.

        The backend is asked first on purpose — if it raises, the cached
        size and buffer are left intact rather than thrown away in
        favour of nothing.
        """
        self._backend.refresh()
        self._screensize = None
        self._img = None
        # Keyed by region shape, so a geometry change makes every cached
        # blur buffer stale as well.
        self._blur_scratch.clear()

    @staticmethod
    def _as_pixels(name, value, bbox):
        """
        Return ``value`` as a plain :class:`int` count of device pixels

        Anything that is exactly a whole number is accepted: :class:`int`,
        ``numpy.int64`` and friends, and integral floats such as ``7.0``
        (Python 3 division produces those readily, and they were already
        usable on the Windows backend).

        Fractional values are rejected rather than rounded — there is no
        honest choice between floor, round and ceil, and any of them
        silently returns a region the caller did not ask for.

        The return value is deliberately a plain ``int``: backends do
        pointer arithmetic with these numbers, and a ``numpy`` integer
        both propagates into ``ctypes`` calls that reject it and wraps
        silently on overflow instead of growing.
        """
        if isinstance(value, bool):
            # bool is an Integral, but a boolean pixel count is always a
            # mistake — and False would quietly mean zero.
            msg = (
                'bbox {} must be a number of device pixels, got the '
                'boolean {!r}.\n'
                'bbox={}'
            ).format(name, value, bbox)
            raise ValueError(msg)

        if isinstance(value, numbers.Integral):
            return int(value)

        if isinstance(value, numbers.Real):
            # Compare against int(value) directly instead of going
            # through float(): float() silently rounds a near-integral
            # Fraction to a whole number, and raises OverflowError on a
            # large one, which would escape as the wrong exception type.
            # int() is exact for every Real, and != then compares
            # exactly.
            try:
                as_int = int(value)
            except (ValueError, OverflowError, TypeError):
                # nan raises ValueError, inf raises OverflowError.
                msg = (
                    'bbox {} must be a finite number of device pixels, '
                    'got {!r}.\n'
                    'bbox={}'
                ).format(name, value, bbox)
                raise ValueError(msg)
            if value != as_int:
                msg = (
                    'bbox {} must be a whole number of device pixels, got '
                    '{!r} — there is no half pixel to capture. Round it '
                    'yourself to say which pixel you mean.\n'
                    'bbox={}'
                ).format(name, value, bbox)
                raise ValueError(msg)
            return as_int

        msg = (
            'bbox {} must be a number of device pixels, got {!r} of type '
            '{}.\n'
            'bbox={}'
        ).format(name, value, type(value).__name__, bbox)
        raise ValueError(msg)

    def check_bbox(self, bbox):
        """
        Validate a bounding box and return it normalized

        Raises :class:`ValueError` if the box is malformed or reaches
        outside the screen bounds. All four components are **device
        pixels**: ``x`` and ``y`` must not be negative, ``width`` and
        ``height`` must be positive, and each must be a whole number
        (see :meth:`_as_pixels` for what counts as one).

        Validating here keeps every backend from having to defend itself
        against a malformed box, which they otherwise report as
        confusing low-level errors — or, on X11, not at all: a negative
        origin reaches ``XGetImage`` as a ``BadMatch`` that the default
        Xlib error handler turns into an outright process exit.

        :param bbox: (x0, y0, width, height) in device pixels.
        :return: the same box as a tuple of four plain :class:`int`
         values, safe to hand to a backend.
        """
        try:
            components = tuple(bbox)
        except TypeError:
            msg = (
                'bbox must be a sequence of four values '
                '(x, y, width, height), got {!r}.'
            ).format(bbox)
            raise ValueError(msg)

        if len(components) != 4:
            msg = (
                'bbox must have exactly four components '
                '(x, y, width, height), got {}.\n'
                'bbox={}'
            ).format(len(components), bbox)
            raise ValueError(msg)

        x, y, w, h = (
            self._as_pixels(name, value, bbox)
            for name, value in zip(('x', 'y', 'width', 'height'), components)
        )

        if x < 0 or y < 0:
            msg = (
                'bbox x and y must not be negative, got x={}, y={}.\n'
                'bbox={}'
            ).format(x, y, bbox)
            raise ValueError(msg)

        if w <= 0 or h <= 0:
            msg = (
                'bbox width and height must be positive, got width={}, '
                'height={} — an empty region has nothing to capture.\n'
                'bbox={}'
            ).format(w, h, bbox)
            raise ValueError(msg)

        if (x + w) > self.screensize[0] or (y + h) > self.screensize[1]:
            msg = (
                'bbox is outside the screen boarders.\n'
                'bbox={} screen size={}'
            ).format(bbox, self.screensize)
            raise ValueError(msg)

        return (x, y, w, h)

    def capture(self, bbox: tuple=None, blur=None) -> numpy.ndarray:
        """
        Take a screenshot and return the image

        The captured image is a height x width for each B, G, R, A channel
        on little-endian Linux x86_64. The alpha channel is typically zeroed.

        # in this example, a full screen screenshot is taken and displayed with
        # matplotlib. Matplotlib is not required and is used for demonstration
        # porposes. Once the image is captured in img that is a numpy array
        # other third party libraries such as opencv can be used to display it
        # quickly with high refresh rates
        .. code-block:: python

            from fastgrab import screenshot
            grab = screenshot.Screenshot()
            img = grab.capture()

            from matplotlib import pyplot as plt
            # matplotlib expects RGB, so reverse the BGR channels and drop alpha
            plt.imshow(img[:, :, 2::-1], interpolation='none')
            plt.show()

        :param bbox: the upper left corner of the screenshot and the width
         and heigh (x0, y0, width, height), in **device pixels**. Each
         component must be a whole number — ``int``, a numpy integer, or
         an integral float such as ``7.0``, which is coerced. A
         fractional value such as ``7.5`` raises :class:`ValueError`;
         round it yourself to say which pixel you mean. ``x``/``y`` must
         not be negative and ``width``/``height`` must be positive.
        :param blur: regions to obscure in this capture, overriding the
         ones passed to the constructor. A list of screen-absolute
         ``(x, y, width, height)`` rectangles, ``True`` for the whole
         frame, or ``False`` / ``[]`` to capture this frame unmodified.
         ``None`` (default) falls back to ``self.blur``. Rectangles are
         clipped to the captured region, and the blur is applied in place
         to the returned buffer — it does not accumulate across calls
         because the backend overwrites the whole buffer every time.
        :return: The image as a numpy array of shape (height, width, 4) in
         BGRA byte order.
        """

        if self._closed:
            raise RuntimeError(
                "this Screenshot has been closed; construct a new one"
            )

        # Resolve the blur before anything is captured: capture() hands
        # back the reused internal buffer, so if an invalid override
        # raised *after* the backend had written into it, a caller still
        # holding a redacted frame from the previous call would find it
        # turned into a clear capture by the very call that failed.
        regions = self.blur if blur is None else _normalise_blur(blur)

        # check/set the dimensions of the image that will be captured
        if bbox is None:
            width, height = self.screensize
            bbox = (0, 0, width, height)

        # Normalized to plain ints — backends do pointer arithmetic with
        # the origin, so a numpy integer or an integral float must not
        # reach them unconverted.
        x, y, width, height = self.check_bbox(bbox)

        bpp = self._backend.bytes_per_pixel()

        # declare the img array only when the image size changes
        if self._img is None:
            self._img = numpy.zeros(
                (height, width, bpp), 'uint8'
            )
        else:
            img_h, img_w = self._img.shape[0:2]
            if img_h != height or img_w != width:
                self._img = numpy.zeros(
                    (height, width, bpp), 'uint8'
                )

        self._backend.screenshot(x, y, self._img)

        if regions:
            # Imported here rather than at module level so a capture that
            # never blurs doesn't pay to import the module at all.
            from fastgrab.effects import blur_regions
            blur_regions(
                self._img,
                None if regions is True else regions,
                style=self.blur_style,
                # The normalized origin from check_bbox, not bbox[0:2]:
                # the raw box may hold a numpy integer or an integral
                # float, and those must not reach the clip arithmetic.
                origin=(x, y),
                scratch=self._blur_scratch,
            )

        return self._img

    def close(self):
        """Release the backend's OS resources and drop the image buffer.

        Optional. Every backend that holds an OS-level frame buffer — the
        wlr SHM buffer, the Windows DIBSection and its memory DC — keeps
        it in a helper object carrying a :mod:`weakref` finalizer, so
        simply dropping a ``Screenshot`` releases the same resources.
        What ``close`` adds is the *moment*: the release happens here
        rather than whenever the garbage collector gets to it.

        That matters most for the shortest form of this library's API,
        which throws the object away by construction::

            img = screenshot.Screenshot().capture()

        In a loop that is a new backend, and a new frame buffer, every
        iteration. CPython frees each one promptly on refcount zero, but
        code holding instances in a container, a cycle, or another
        implementation's deferred collector will not. Reuse one
        ``Screenshot``, or close what you finish with::

            with screenshot.Screenshot() as grab:
                img = grab.capture()

        Idempotent, and safe on a backend that holds nothing. The
        instance must not be used for capture afterwards.
        """
        self._closed = True
        self._img = None
        # These outweigh the frame buffer — a full-frame 1080p blur holds
        # several float32 planes — so releasing _img without them would
        # miss most of what close() is for.
        self._blur_scratch.clear()
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
