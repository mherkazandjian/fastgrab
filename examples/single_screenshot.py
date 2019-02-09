"""
take a single screenshot
"""
from fastgrab import screenshot

grab = screenshot.Screenshot()
img = grab.capture(bbox=(0, 0, 3840, 2160))

