"""
example script that uses the C api call to take a screenshot.

This is handy to avoid the overhead of high-level python wrapper
"""
import numpy
from fastgrab._linux_x11 import screenshot

# a full HD screen
x, y, width, height = 0, 0, 1920, 1080
img = numpy.zeros((height, width, 4), 'uint8')
screenshot(x, y, img)

# (optional) view the screenshot
# import pylab
# matplotlib expects RGB, so reverse the BGR channels and drop alpha
# pylab.imshow(img[:, :, 2::-1], interpolation='none')
# pylab.show()
