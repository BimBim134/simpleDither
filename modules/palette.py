import numpy as np

PICO8 = np.array([[
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

FOUR = np.array([[[32, 26, 102],
                      [191, 59, 38],
                      [76, 230, 46],
                      [255, 255, 255]]]) / 255

BW = np.array([[[0, 0, 0],
                      [255, 255, 255]]]) / 255

PERLES = np.array([[
    [0, 0, 0], # black
    [10, 144, 57], # green
    [12, 120, 185], # blue  
    [170, 108, 181], # purple
    [242, 121, 198], # pink
    [255, 255, 255], # white
    [221, 179, 61], # yellow  
    [235, 107, 72], # orange
    [208, 57, 64], # red
    [107, 41, 51] # brown
]]) /255

RGBY = np.array([
    [[255, 0, 0],   # Red
     [0, 255, 0],   # Green
     [0, 0, 255],   # Blue
     [255, 255, 0]], # Yellow
]) /255