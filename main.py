#!/usr/bin/env python
import numpy as np
from multiprocessing import Pool

import modules.image_processing as dither
import modules.palette as pal

img_path = 'IN/DSCF9150.JPG'

img = dither.dimg(img_path)
img.palette = dither.GAMEBOY
img.square_crop()
img.resize(target_width=140)
img.bayer(16)
# img.fs()
# img.atk()
# img.stucki()
# img.burkes()
# img.sierra()
img.save()


exit()

todo = [
    ['IN/DSCF7667.JPG',(380, 380), pal.BW,'simple',   16, 0.5]
]

if __name__ == '__main__':
    with Pool() as p:
        p.map(dither.processPicture_paralel, todo)
    