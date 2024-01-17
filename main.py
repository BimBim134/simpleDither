#!/usr/bin/env python
import numpy as np
from multiprocessing import Pool

import modules.image_processing as dither
import modules.palette as pal


todo = [
    ['IN/DSCF7667.JPG',(380, 380), pal.BW,'simple',   16, 0.5]
]

if __name__ == '__main__':
    with Pool() as p:
        p.map(dither.processPicture_paralel, todo)
    