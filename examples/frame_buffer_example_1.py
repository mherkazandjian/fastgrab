"""
<keywords>
test, python, fastgrab, frame, buffer, framebuffer
</keywords>
<description>
A minimal example of using the frame buffer class
</description>
<seealso>
</seealso>
"""

from __future__ import print_function
from fastgrab.dataio import FrameBuffer
from pandas import Timestamp, Series, DataFrame, Panel4D
import numpy

fb = FrameBuffer(dtype=[('timestamp', 'S27'), ('frame', '(3, 20, 20)i4')])

element = fb.data.copy()

for i in range(10):
    element['timestamp'] = str(Timestamp('now'))
    element['frame'] = numpy.ones((3, 20, 20), 'i4') + i
    fb.append(element)

timestamps = Series([Timestamp(t[0]) for t in fb.data['timestamp'][0:10]])
values = fb.data['frame'][0:10].reshape(10, 3, 20, 20)

data = Panel4D(values, labels=timestamps)

print('done')
