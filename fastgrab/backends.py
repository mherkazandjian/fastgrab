"""
<keywords>
python, backends, screenshot
</keywords>
<description>
backends to various wrapper that wrap libraries and commands to take
screenshots
</description>
<seealso>
</seealso>
"""
import os
from enum import Enum, unique
from numpy import zeros
from scipy.misc import imread
from ctypes import cdll, c_int32, c_uint8, POINTER

from fastgrab.utils import logger

try:
    import autopy
except ImportError as e:
    logger.info('could not import autopy')
    autopy = None

try:
    import linux_x11
except ImportError as e:
    logger.info('could not import linux_x11')
    linux_x11 = None


__all__ = [
    'Backends',
    'AutopyBackend',
    'OsxBackend',
    'LinuxX11Backend',
    'VlcX11Backend',
    'BackendFactory',
]


class BackendBase(object):
    """
    The base class for backends to take screenshots.
    """
    def __init__(self, *args, **kwargs):
        """
        constructor
        """
        self._available = None

    def is_available(self):
        """Returns True if the backend is available, otherwise returns False
        :return: bool
        """
        raise NotImplementedError('''must be implemented by subclass''')

    @property
    def available(self):
        """getter"""
        return self._available

    @available.setter
    def available(self, value):
        """setter"""
        self._available = value

    def screenshot(self):
        """
        take a screenshot and return it as a numpy array
        :return: array-like
        """
        raise NotImplementedError('''must be implemented by subclass''')


class AutopyBackend(BackendBase):
    """
    backend that wraps the functionality of some of the routines of autopy
    """
    def __init__(self, *args, **kwargs):
        """
        constructor
        """
        super(AutopyBackend, self).__init__(*args, **kwargs)

    def is_available(self):
        """
        .. todo:: inherit the docstring from parent class
        """
        if autopy is not None:
            return True
        else:
            return False

    def screenshot(self, rect):
        """
        take a screenshot using autopy. Currently this method requires that
        the partition /dev/shm is writable, since autopy can not return the
        raw image buffer.
        .. see examples/screencap_example_2.py
        for a description on how to do that (i have not figured it out fully
        by there it is mentioned where to look

        :param list rect: a list that specifies the range on the display where
         the screenshot will be taken. (x, y, width, height)
        :return: a numpy array of the 3 channels of the image
        """
        img = autopy.bitmap.capture_screen(((rect[0], rect[1]),
                                            (rect[2], rect[3])))
        # .. todo:: get the tmp file name from something like this
        # .. todo:: os.path.join(config.tmpdir.screenshots, 'autopy.tmp.bmp')
        tmpfile = os.path.join('/dev/shm/autopy.tmp.bmp')
        img.save(tmpfile)

        # import mmap
        # m = mmap.mmap(-1, 2 << 30)
        # asdads
        # img.save()
        # import pdb
        # pdb.set_trace()
        #

        return imread(open(tmpfile, 'r'))


class OsxBackend(BackendBase):
    """

    :param BackendBase:
    :return:
    """
    def __init__(self):
        """

        :param self:
        :return:
        """
        raise NotImplementedError('''''')

    def screenshot(self):
        """

        :return:
        """
        raise NotImplementedError('''''')


class LinuxX11Backend(BackendBase):
    """
    Backend that uses the linux_x11 module to call the screenshot (C) function.

    .. code-block:: python

         backend = LinuxX11Backend()
         img = backend.screenshot(0, 0, 100, 100)
         import pylab
         pylab.imshow(img)
         pylab.show()

    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super(LinuxX11Backend, self).__init__(*args, **kwargs)

        self.interface = None
        """the interface to the C function screenshot in linux_x11"""

        self._available = self.is_available()

        if self._available is True:
            self.interface = cdll.LoadLibrary(linux_x11.__file__).screenshot
            self.interface.argtypes = [
                c_int32, c_int32, c_int32, c_int32,
                POINTER(c_uint8)
            ]

    def is_available(self):
        """
        .. todo:: inherit the docstring from parent class
        """
        if linux_x11 is not None:
            return True
        else:
            return False

    def load_library(self):
        """
        Load the linux_x11.so shared library using the ctypes module and returns
         it.
        :return: ctypes shared library object
        """
        library = cdll.LoadLibrary(linux_x11.__file__)
        return library

    def setup_interface_function(self):
        """
        sets up the screenshot function in the linux_x11 library and returns
        it
        :return: callable: C function interfrace that takes a screenshot
        """
        library = self.load_library()
        interface_func = library.screenshot
        interface_func.artypes = [c_int32, c_int32, c_int32, c_int32,
                                  POINTER(c_uint8)]
        return interface_func

    def screenshot(self, x, y, width, height):
        """
        take a screenshot using the linux_x11 screenshot library and returns
        a numpy array
        :param int x: The pixel position from the left.
        :param int y: The pixel position from the top.
        :param int width: The width of the region.
        :param int height: The height of the region.
        :return: Numpy array of dtype uint8 with shape width,height,3
        """
        img = zeros((height, width, 4), 'uint8')
        img_data_ptr = img.ctypes.data_as(POINTER(c_uint8))
        self.interface(x, y, img.shape[1], img.shape[0], img_data_ptr)
        return img[:, :, 0:3]


class VlcX11Backend(BackendBase):
    """
    vlc backend via python-vlc

         ~> pip install python-vlc

    :param BackendBase:
    :return:
    """
    def __init__(self):
        """

        :param self:
        :return:
        """
        raise NotImplementedError('''''')

    def screenshot(self):
        """

        :return:
        """
        raise NotImplementedError('''''')


@unique
class Backends(Enum):
    """
    Enumerator for implemented backends
    """
    osx = OsxBackend
    linux_x11 = LinuxX11Backend
    autopy = AutopyBackend
    vlc = VlcX11Backend


class BackendFactory(object):
    """

    """
    def use(self):
        """

        :return:
        """
        raise NotImplementedError('''''')
