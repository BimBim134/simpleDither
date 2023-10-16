#!/usr/bin/env python
import numpy as np
from multiprocessing import Pool

import modules.image_processing as dither
import modules.palette as pal

dither.save_color_palette(pal.PERLES, 'IN/PERLES.png')

todo = [
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'bayer',2],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'bayer',4],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'bayer',8],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'bayer',16],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'fs'],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'atk'],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'jjn'],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'simple'],
    ['IN/DSCF6798.JPG',(128, 128), pal.PERLES,'closest'],
]

if __name__ == '__main__':
    with Pool() as p:
        p.map(dither.processPicture_paralel, todo)