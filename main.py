#!/usr/bin/env python
import numpy as np
from multiprocessing import Pool

import modules.image_processing as dither
import modules.palette as pal

dither.processPicture('IN/DSCF6798.JPG',(128, 128), pal.PICO8, 'bayer', 8)

exit()

todo = [
    ['IN/DSCF6301.jpg',(128, 128), pal.PERLES,'atk'],
    ['IN/DSCF6301.jpg',(128, 128), pal.PERLES,'jjn'],
    ['IN/DSCF6301.jpg',(128, 128), pal.PERLES,'fs'],
    ['IN/DSCF6301.jpg',(128, 128), pal.PERLES,'simple'],

    ['IN/DSCF6798.jpg',(128, 128), pal.PERLES,'atk'],
    ['IN/DSCF6798.jpg',(128, 128), pal.PERLES,'jjn'],
    ['IN/DSCF6798.jpg',(128, 128), pal.PERLES,'fs'],
    ['IN/DSCF6798.jpg',(128, 128), pal.PERLES,'simple'],

    ['IN/DSCF6852.jpg',(128, 128), pal.PERLES,'atk'],
    ['IN/DSCF6852.jpg',(128, 128), pal.PERLES,'jjn'],
    ['IN/DSCF6852.jpg',(128, 128), pal.PERLES,'fs'],
    ['IN/DSCF6852.jpg',(128, 128), pal.PERLES,'simple'],
]

if __name__ == '__main__':
    with Pool(5) as p:
        p.map(dither.processPicture_paralel, todo)