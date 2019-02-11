"""
take a single full-screen screenshot
"""
from fastgrab import screenshot
img = screenshot.Screenshot().capture()

# import pylab
# pylab.imshow(img[:, :, 0:3], interpolation='none', cmap='Greys_r')
# pylab.show()

