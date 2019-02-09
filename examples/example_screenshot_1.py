"""
<keywords>
test, python, screenshot, fastgrab
</keywords>
<description>
A minimal example of taking one screenshot using fastgrab
</description>
<seealso>
</seealso>
"""
from __future__ import print_function
from fastgrab.backends import AutopyBackend
import pylab

backend = AutopyBackend()
img = backend.screenshot([0, 0, 500, 500])
pylab.imshow(img)
pylab.show()
print('done')
