"""
eyeball every blur method at once

Captures the screen once, applies each method to a copy, tiles the
results into a 2x2 grid and writes it as a BMP you can just open. Uses
the standard library only — no Pillow, no matplotlib — so it runs on a
bare ``pip install fastgrab``.

    python examples/blur_gallery.py [output.bmp]
"""
import struct
import sys

import numpy

from fastgrab import screenshot
from fastgrab.effects import BlurStyle, blur_regions


def write_bmp(path, img):
    """Write a BGRA numpy array as a 24-bit BMP.

    BMP stores rows bottom-up as B, G, R — which is exactly fastgrab's
    channel order minus the alpha, so this is a slice and a flip.
    """
    height, width = img.shape[:2]
    rows = numpy.ascontiguousarray(img[::-1, :, :3])
    padding = (-width * 3) % 4
    stride = width * 3 + padding
    pixels = bytearray()
    for row in rows:
        pixels += row.tobytes() + b"\x00" * padding
    header = struct.pack(
        "<2sIHHI", b"BM", 14 + 40 + len(pixels), 0, 0, 14 + 40
    )
    info = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 24, 0, stride * height,
        2835, 2835, 0, 0,
    )
    with open(path, "wb") as fobj:
        fobj.write(header + info + pixels)


def main(argv):
    out = argv[1] if len(argv) > 1 else "blur_gallery.bmp"

    grab = screenshot.Screenshot()
    width, height = grab.screensize
    # A quarter of the screen, rounded even, so the 2x2 grid is tidy.
    tile_w = (width // 2) & ~1
    tile_h = (height // 2) & ~1
    frame = grab.capture(bbox=(0, 0, tile_w, tile_h)).copy()
    print("captured {}x{} from a {}x{} screen".format(
        tile_w, tile_h, width, height
    ))

    styles = [
        ("box", BlurStyle(method="box", radius=12)),
        ("gaussian", BlurStyle(method="gaussian", radius=12)),
        ("pixelate", BlurStyle(method="pixelate", block=16)),
        ("fill", BlurStyle(method="fill", color=(0, 0, 0))),
    ]

    # Blur the middle half of each tile so you can see the boundary: the
    # blur must stop dead at the rectangle's edge, with no smearing.
    region = (tile_w // 4, tile_h // 4, tile_w // 2, tile_h // 2)

    gallery = numpy.zeros((tile_h * 2, tile_w * 2, 4), numpy.uint8)
    for index, (name, style) in enumerate(styles):
        tile = frame.copy()
        blur_regions(tile, [region], style)
        row, col = divmod(index, 2)
        gallery[row * tile_h:(row + 1) * tile_h,
                col * tile_w:(col + 1) * tile_w] = tile
        print("  {:<9s} region {} applied".format(name, region))

    write_bmp(out, gallery)
    print("wrote {} ({}x{})  — top-left box, top-right gaussian,".format(
        out, tile_w * 2, tile_h * 2
    ))
    print("     bottom-left pixelate, bottom-right fill")


if __name__ == "__main__":
    main(sys.argv)
