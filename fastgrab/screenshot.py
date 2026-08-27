"""
Module that implements the object for taking screenshots
"""
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
        if len(blur) == 0:
            # Documented as "capture this frame unmodified", so it must
            # not drag in fastgrab.effects. Only a sized container can be
            # checked without consuming it; an empty generator still
            # imports, which is a fair price for not eating its contents.
            return ()
    except TypeError:
        pass
    from fastgrab.effects import _normalise_regions
    return _normalise_regions(blur)


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

    def check_bbox(self, bbox):
        """
        Raise an exception of the bounding box is outside the screen bounds
        """
        x, y, w, h = bbox
        if (x + w) > self.screensize[0] or (y + h) > self.screensize[1]:
            msg = (
                'bbox is outside the screen boarders.\n'
                'bbox={} screen size={}'
            ).format(bbox, self.screensize)
            raise ValueError(msg)

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
            plt.imshow(img[:, :, 0:3], interpolation='none', cmap='Greys_r')
            plt.show()

        :param bbox: the upper left corner of the screenshot and the width
         and heigh (x0, y0, width, height).
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

        # Resolve the blur first: capture() hands back the reused
        # internal buffer, so if an invalid override raised *after* the
        # backend had written into it, a caller still holding a redacted
        # frame from the previous call would find it turned into a clear
        # capture by the very call that failed.
        regions = self.blur if blur is None else _normalise_blur(blur)

        # check/set the dimensions of the image that will be captured
        if bbox is None:
            width, height = self.screensize
            bbox = (0, 0, width, height)
        else:
            _, _, width, height = bbox

        self.check_bbox(bbox)

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

        self._backend.screenshot(bbox[0], bbox[1], self._img)

        if regions:
            # Imported here rather than at module level so a capture that
            # never blurs doesn't pay to import the module at all.
            from fastgrab.effects import blur_regions
            blur_regions(
                self._img,
                None if regions is True else regions,
                style=self.blur_style,
                origin=(bbox[0], bbox[1]),
                scratch=self._blur_scratch,
            )

        return self._img
