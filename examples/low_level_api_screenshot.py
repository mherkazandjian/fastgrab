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
# pylab.imshow(img[:, :, 0:3], interpolation='none', cmap='Greys_r')
# pylab.show()
