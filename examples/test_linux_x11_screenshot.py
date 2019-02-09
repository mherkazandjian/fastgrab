"""
<keywords>
test, python, shared, library, screenshot
</keywords>
<description>
example script that uses the shared library built with distutils to take a
 screenshot
</description>
<seealso>
</seealso>
"""
import time
# import pylab
from fastgrab.backends import LinuxX11Backend

# vertical FHD screen
x, y, w, h = 0, 0, 1080, 1920
rect = (x, y, w, h)

# horizontal 4K screen to the right of a vertical FHD screen
#x, y, w, h = 1080, 0, 2*1920, 2*1080
#rect = (x, y, w, h)

backend = LinuxX11Backend()

n = 100
t0 = time.time()
for i in range(n):
    screenshot = backend.screenshot(x, y, h, w)
print(n / (time.time() - t0))

# pylab.imshow(screenshot, interpolation='none', cmap='Greys_r')
# pylab.show()
print('done')
