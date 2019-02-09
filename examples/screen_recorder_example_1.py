from __future__ import print_function

import os
import pylab
import numpy
import time
import cv2
import pdb

from pandas import Timestamp

from fastgrab.dataio import FrameBuffer
from fastgrab.backends import (AutopyBackend, LinuxX11Backend)


# backend = AutopyBackend()
backend = LinuxX11Backend()
region = (1080, 0, 400, 400)

fb = FrameBuffer(dtype=[('timestamp', 'S27'), ('frame', '(400, 400, 3)uint8')])
element = fb.data.copy()

def play(data):

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0, 20)

    for datum in data:

        img = datum['frame'][0]
        timestamp = datum['timestamp'][0]

        cv2.putText(img, timestamp, position, font, fontScale=0.5,
                    color=(255, 0, 0), thickness=1)
        cv2.imshow("Frame", img)
        cv2.waitKey(30)

for i in range(100):

    time.sleep(0.01)

    img = backend.screenshot(region)

    element['timestamp'] = str(Timestamp('now'))
    element['frame'] = img
    fb.append(element)

    print('{}, frame num {}'.format(element['timestamp'][0], i))

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0, 20)

    img_tmp = img.copy()

    timestamp = element['timestamp'][0]
    cv2.putText(img_tmp, timestamp, position, font, 0.5, (255, 0, 0))

    cv2.imshow("Frame", img_tmp)
    cv2.waitKey(1)

# saving the buffer data
print('saving the screen recording...')
saved_data_path = os.path.expanduser('~/tmp/sandbox/foo.npz')
fb.save(saved_data_path)
print('\t\tsaved')

print('\nplaying the recording...')
fb_new = FrameBuffer()
fb_new.load(saved_data_path)
play(fb_new.data)

print('done')
