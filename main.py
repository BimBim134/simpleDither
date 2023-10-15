#!/usr/bin/env python

import numpy as np

import modules.image_processing as dither

pico8_palette = np.array([[
    [0, 0, 0],           # Black
    [29, 43, 83],        # Dark Blue
    [126, 37, 83],       # Dark Purple
    [0, 135, 81],        # Dark Green
    [171, 82, 54],       # Brown
    [95, 87, 79],        # Dark Gray
    [194, 195, 199],     # Light Gray
    [255, 241, 232],     # White
    [255, 0, 77],        # Red
    [255, 163, 0],       # Orange
    [255, 236, 39],      # Yellow
    [0, 228, 54],        # Green
    [41, 173, 255],      # Blue
    [131, 118, 156],     # Indigo
    [255, 119, 168],     # Pink
    [255, 204, 170]      # Peach
]]) /255

palette = np.array([[[32, 26, 102],
                      [191, 59, 38],
                      [76, 230, 46],
                      [255, 255, 255]]]) / 255

dither.processPicture('/home/bimbim/Codes/simpleDither/IN/DSCF6852.JPG',
                      (128, 128),
                      pico8_palette,
                      'atk')

dither.processPicture('/home/bimbim/Codes/simpleDither/IN/DSCF6852.JPG',
                      (128, 128),
                      pico8_palette,
                      'jjn')

dither.processPicture('/home/bimbim/Codes/simpleDither/IN/DSCF6852.JPG',
                      (128, 128),
                      pico8_palette,
                      'fs')

dither.processPicture('/home/bimbim/Codes/simpleDither/IN/DSCF6852.JPG',
                      (128, 128),
                      pico8_palette,
                      'simple')
