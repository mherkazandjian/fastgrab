"""
blur and redact regions of a screenshot

Two ways to do it: pass ``blur=`` to ``capture()``, or call
``blur_regions`` yourself on any BGRA array you already have.
"""
from fastgrab import screenshot
from fastgrab.effects import BlurStyle, blur_regions

grab = screenshot.Screenshot(
    blur_style=BlurStyle(method='gaussian', radius=16)
)
width, height = grab.screensize
print('screen: {}x{}'.format(width, height))

# regions are (x, y, width, height) in screen coordinates
soft = (0, 0, min(320, width), min(160, height))
secret = (0, min(200, height - 1), min(320, width), min(80, height))

# 1. blur one region while capturing — the returned buffer is already
#    blurred, using the style the Screenshot was constructed with
img = grab.capture(blur=[soft])
print('captured {} {}'.format(img.shape, img.dtype))

# 2. redact a second region afterwards with an opaque box. Only 'fill'
#    actually destroys the pixels — use it for passwords and tokens,
#    a blur only softens them.
blur_regions(img, [secret], BlurStyle(method='fill', color=(0, 0, 0)))

print('blurred region  top-left pixel BGRA:', img[0, 0])
print('redacted region top-left pixel BGRA:', img[secret[1], secret[0]])

# import pylab
# pylab.imshow(img[:, :, 2::-1], interpolation='none')
# pylab.show()
