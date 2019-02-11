"""
take a single full-screen screenshot
"""
import time
from fastgrab import screenshot

# the resolutions to be benchmarked
benchmark_resolutions = {
    '360p': (480, 360),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4K': (1920*2, 1080*2),
    '8K': (1920*4, 1080*4)
}

# the number of frames to be captured for each benchmark resolution
n_frames = 800

# run the benchmark
grab = screenshot.Screenshot()

print('benchmark capture frame rate using frames')
print(f'the benchmark will capture {n_frames} frames\n')
print('resolution standard | resolution    |  measured fps')
print('--------------------+---------------+--------------')

for resolution_standard, resolution in benchmark_resolutions.items():

    print('{:<8s}            |  {:<10s}   |'.format(
        resolution_standard,
        '{:d}x{:d}'.format(resolution[0], resolution[1])
    ), end='')
    try:
        t0 = time.time()
        for i in range(n_frames):
            img = grab.capture(bbox=(0, 0, resolution[0], resolution[1]))
        tf = time.time()
        fps = n_frames / (tf - t0)
        print('  {:.1f}   '.format(fps))
    except ValueError as exc:
        print(f'  failed')
