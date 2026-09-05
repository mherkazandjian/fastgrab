"""
take a single full-screen screenshot
"""
from fastgrab import screenshot
img = screenshot.Screenshot().capture()

# import pylab
# matplotlib expects RGB, so reverse the BGR channels
# pylab.imshow(img[:, :, 2::-1], interpolation='none')
# pylab.show()

