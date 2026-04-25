"""
Module that implements the object for taking screenshots
"""
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
            plt.imshow(img[:, :, 0:3], interpolation='none', cmap='Greys_r')
            plt.show()

        :param bbox: the upper left corner of the screenshot and the width
         and heigh (x0, y0, width, height).
        :return: The image as a numpy array of shape (height, width, 4) in
         BGRA byte order.
        """

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

        return self._img
