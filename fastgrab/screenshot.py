"""
Module that implements the object for taking screenshots
"""
import numbers

import numpy

from fastgrab.backends import _resolve_backend


class Screenshot(object):
    """
    Main object that captures screenshots and provides other utilities
    """
    def __init__(self, backend=None):
        """
        Constructor

        :param backend: optional explicit backend name — one of
            ``'x11'``, ``'wlr'``, ``'portal'``. When ``None`` (default)
            the backend is auto-detected from the environment:
            Wayland sessions try wlr → portal; X11 sessions use x11.
        """
        self._backend = _resolve_backend(backend)
        """The capture backend (BaseBackend subclass instance)"""

        self._screensize = None
        """tuple, (width, height), backing variable for self.screensize"""

        self._img = None
        """The buffer where the captured image is stored"""

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

    def capture(self, bbox: tuple=None) -> numpy.ndarray:
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
        :return: The image as a numpy array of shape (height, width, 4) in
         BGRA byte order.
        """

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

        return self._img
