"""
Module that implements the object for taking screenshots
"""
import numpy

class Screenshot(object):
    """
    Main object that captures screenshots and provides other utilities
    """
    def __init__(self):
        """
        Constructor
        """

        self._screensize = None
        """tuple, (width, height), backing variable for self.screensize"""

        self._img = None

    def _find_screensize_using_tk(self):
        """

        :return:
        """
        pass

    def _find_screensize_using_xrandr(self):
        """

        :return:
        """
        pass

    @property
    def screensize(self) -> tuple:
        """
        return the screensize/resolution
        """
        if self._screensize is None:
            try:
                self._screensize = None  # fetch the screensize from tk
            except:
                pass
        return None, None

    def capture(self, bbox: tuple=None, save: bool=False) -> numpy.zeros:
        """
        Take a screenshot and return the image

        :param bbox:
        :param save:
        :return:
        :return:
        """

        # check/set the dimensions of the image that will be captured
        if bbox is None:
            width, height = self.screensize
        else:
            _, _, width, height = bbox

        # declare the img array only when the image size changes
        if self._img is None:
            self._img = numpy.zeros((height, width, 4), 'uint8')

        # interface(x, y, img.shape[1], img.shape[0], img_data_ptr)
        # return img[:, :, 0:3]
