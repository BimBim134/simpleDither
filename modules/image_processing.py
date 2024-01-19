from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image
from skimage import io
from skimage.transform import resize
from skimage.util import crop

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

BW = np.array([[[0, 0, 0],
                 [255, 255, 255]]]) / 255

RGB = np.array([[
    [0, 0, 0], 
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 0],
    [255, 255, 255]
]]) /255


MEL = np.array([[
    [0, 0, 0], 
    [231, 70, 69],
    [251, 119, 86],
    [250, 205, 96],
    [253, 250, 102],
    [26, 192, 198],
    [255, 255, 255]
]]) /255


@njit(cache=True)
def squareCropCoordinate(image):
    # return the coordinate of the largest square
    # possible in the image
    height = image.shape[0]
    width = image.shape[1]
    # is the image already square ?
    if height == width:
        return ((0, 0), (0, 0), (0, 0))
    # Is the image in portrait or landscape mode
    is_portrait = height > width
    if is_portrait:
        deltaX = int((height - width) / 2)
        return ((deltaX, deltaX), (0, 0), (0, 0))
    else:
        deltaY = int((width - height) / 2)
        return ((0, 0), (deltaY, deltaY), (0, 0))


@njit(cache=True)
def findClosest(value, palette, type='euclidian'):
    x = palette[0, :, 0]
    y = palette[0, :, 1]
    z = palette[0, :, 2]
    dx = x - value[0]
    dy = y - value[1]
    dz = z - value[2]
    if type == 'euclidian':
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        dist = np.abs(dx) + np.abs(dy) + np.abs(dz)
    return palette[0, np.argmin(dist), :]


@njit(cache=True)
def dithering_atk(image, palette):
    image = np.concatenate(
        (np.zeros((image.shape[0], 1, 3)), image,
         np.zeros((image.shape[0], 2, 3))),
        axis=1)
    image = np.concatenate(
        (image, np.zeros((2, image.shape[1], 3))),
        axis=0)
    output = np.copy(image)
    for y in np.arange(1, output.shape[1] - 2, 1):
        for x in np.arange(0, output.shape[0] - 2, 1):
            oldPixel = np.copy(output[x, y, :])
            newPixel = findClosest(oldPixel, palette)
            output[x, y, :] = newPixel
            # atkinson algorithm
            # . * 1 1
            # 1 1 1 .       x 1/6
            # . 1 . .
            quant_err = (oldPixel - newPixel) / 6
            output[x+1, y, :] = output[x+1, y, :] + quant_err
            output[x+2, y, :] = output[x+2, y, :] + quant_err
            output[x-1, y+1, :] = output[x-1, y+1, :] + quant_err
            output[x, y+1, :] = output[x, y+1, :] + quant_err
            output[x+1, y+1, :] = output[x+1, y+1, :] + quant_err
            output[x, y+2, :] = output[x, y+2, :] + quant_err
    # crop the un-converged pixels
    output = output[:-2, 1:-2, :]
    # clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))
    return output



@njit(cache=True)
def dithering_jjn(image, palette):
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (np.zeros((image.shape[0], 2, 3)), image,
         np.zeros((image.shape[0], 2, 3)),
         np.zeros((image.shape[0], 1, 3))),
        axis=1)
    image = np.concatenate(
        (image, np.zeros((2, image.shape[1], 3))),
        axis=0)
    output = np.copy(image)
    for y in range(1, output.shape[1] - 2):
        for x in range(0, output.shape[0] - 2):
            oldPixel = np.copy(output[x, y, :])
            newPixel = findClosest(oldPixel, palette)
            output[x, y, :] = newPixel
            # JJN dithering algorithm
            # . . * 7 5
            # 3 5 7 5 3     * 1/48
            # 1 3 5 3 1
            quant_err = oldPixel - newPixel
            output[x+1, y, :] += quant_err * 7 / 48
            output[x+2, y, :] += quant_err * 5 / 48
            output[x-2, y+1, :] += quant_err * 3 / 48
            output[x-1, y+1, :] += quant_err * 5 / 48
            output[x, y+1, :] += quant_err * 7 / 48
            output[x+1, y+1, :] += quant_err * 5 / 48
            output[x+2, y+1, :] += quant_err * 3 / 48
            output[x-2, y+2, :] += quant_err * 1 / 48
            output[x-1, y+2, :] += quant_err * 3 / 48
            output[x, y+2, :] += quant_err * 5 / 48
            output[x+1, y+2, :] += quant_err * 3 / 48
            output[x+2, y+2, :] += quant_err * 1 / 48
    # Crop the un-converged pixels
    output = output[:-2, 2:-3, :]
    # Clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))
    return output


@njit(cache=True)
def dithering_simple(image, palette):
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (image,
         np.zeros((image.shape[0], 1, 3))),
        axis=1)
    image = np.concatenate(
        (image, np.zeros((1, image.shape[1], 3))),
        axis=0)
    output = np.copy(image)
    for y in range(1, output.shape[1] - 1):
        for x in range(0, output.shape[0] - 1):
            oldPixel = np.copy(output[x, y, :])
            newPixel = findClosest(oldPixel, palette)
            output[x, y, :] = newPixel
            # Simple 2D error diffusion matrix
            quant_err = oldPixel - newPixel
            output[x+1, y, :] += quant_err * 0.5
            output[x, y+1, :] += quant_err * 0.5
    # Crop the un-converged pixels
    output = output[:-1, :-1, :]
    # Clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))
    return output


@njit(cache=True)
def dithering_fs(image, palette):
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (np.zeros((image.shape[0], 1, 3)), image,
         np.zeros((image.shape[0], 1, 3))),
        axis=1)
    image = np.concatenate(
        (image, np.zeros((1, image.shape[1], 3))),
        axis=0)
    output = np.copy(image)
    for y in range(1, output.shape[1] - 1):
        for x in range(0, output.shape[0] - 1):
            oldPixel = np.copy(output[x, y, :])
            newPixel = findClosest(oldPixel, palette)
            output[x, y, :] = newPixel
            # Floyd-Steinberg dithering algorithm
            # . * 7
            # 3 5 1
            quant_err = oldPixel - newPixel
            output[x+1, y, :] += quant_err * 7 / 16
            output[x-1, y+1, :] += quant_err * 3 / 16
            output[x, y+1, :] += quant_err * 5 / 16
            output[x+1, y+1, :] += quant_err * 1 / 16
    # Crop the un-converged pixels
    output = output[:-1, 1:-1, :]
    # Clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))

    return output


@njit(cache=True)
def dithering_Stucki(image, palette):
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (np.zeros((image.shape[0], 3, 3)),
         image,
         np.zeros((image.shape[0], 2, 3))),
        axis=1)
    image = np.concatenate(
        (image,
         np.zeros((2, image.shape[1], 3))),axis=0)
    output = np.copy(image)
    for y in range(0, output.shape[0] - 2):
        for x in range(2, output.shape[1] - 2):
            old_pixel = np.copy(output[y, x, :])
            new_pixel = findClosest(old_pixel, palette)
            output[y, x, :] = new_pixel

            # Stucki algorithm
            #             X   8   4
            # 2   4   8   4   2
            # 1   2   4   2   1

            quant_err = (old_pixel - new_pixel) / 42
            output[y, x + 1, :] = output[y, x + 1, :] + quant_err * 8
            output[y, x + 2, :] = output[y, x + 2, :] + quant_err * 4

            output[y + 1, x - 2, :] = output[y + 1, x - 2, :] + quant_err * 2
            output[y + 1, x - 1, :] = output[y + 1, x - 1, :] + quant_err * 4
            output[y + 1, x, :] = output[y + 1, x, :] + quant_err * 8
            output[y + 1, x + 1, :] = output[y + 1, x + 1, :] + quant_err * 4
            output[y + 1, x + 2, :] = output[y + 1, x + 2, :] + quant_err * 2

            output[y + 2, x - 2, :] = output[y + 2, x - 2, :] + quant_err * 1
            output[y + 2, x - 1, :] = output[y + 2, x - 1, :] + quant_err * 2
            output[y + 2, x, :] = output[y + 2, x, :] + quant_err * 4
            output[y + 2, x + 1, :] = output[y + 2, x + 1, :] + quant_err * 2
            output[y + 2, x + 2, :] = output[y + 2, x + 2, :] + quant_err * 1
    # Crop the un-converged pixels
    output = output[:-2,3:-2,:]
    # Clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))

    return output


@njit(cache=True)
def dithering_Burkes(image, palette):
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (np.zeros((image.shape[0], 3, 3)),
         image,
         np.zeros((image.shape[0], 2, 3))),
        axis=1)
    image = np.concatenate(
        (image,
         np.zeros((1, image.shape[1], 3))),axis=0)
    output = np.copy(image)
    for y in range(0, output.shape[0] - 1):
        for x in range(2, output.shape[1] - 2):
            old_pixel = np.copy(output[y, x, :])
            new_pixel = findClosest(old_pixel, palette)
            output[y, x, :] = new_pixel

            # Burkes algorithm
            quant_err = (old_pixel - new_pixel) / 32
            output[y, x + 1, :] = output[y, x + 1, :] + quant_err * 8
            output[y, x + 2, :] = output[y, x + 2, :] + quant_err * 4

            output[y + 1, x - 2, :] = output[y + 1, x - 2, :] + quant_err * 2
            output[y + 1, x - 1, :] = output[y + 1, x - 1, :] + quant_err * 4
            output[y + 1, x, :] = output[y + 1, x, :] + quant_err * 8
            output[y + 1, x + 1, :] = output[y + 1, x + 1, :] + quant_err * 4
            output[y + 1, x + 2, :] = output[y + 1, x + 2, :] + quant_err * 2

    # Crop the un-converged pixels
    output = output[:-1,3:-2,:]
    # Clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))

    return output



@njit(cache=True)
def generate_bayer_matrix(size):
    if size < 2:
        raise ValueError("Bayer matrix size must be at least 2x2.")
    bayer_matrix = np.zeros((size, size), dtype=np.uint8)
    n = 1
    while n < size:
        for i in range(n):
            for j in range(n):
                bayer_matrix[i][j] *= 4
                bayer_matrix[i + n][j] = bayer_matrix[i][j] + 2
                bayer_matrix[i][j + n] = bayer_matrix[i][j] + 3
                bayer_matrix[i + n][j + n] = bayer_matrix[i][j] + 1
        n *= 2
    return bayer_matrix


@njit(cache=True)
def dithering_bayer(image, palette, matrix_size=4):
    # Get the dimensions of the input image
    image_height, image_width, _ = image.shape
    # Define a Bayer matrix that matches the dimensions of the input image
    bayer_matrix = generate_bayer_matrix(matrix_size)
    # Normalize the Bayer matrix to match the palette size
    bayer_matrix = (bayer_matrix / matrix_size**2) * (palette.shape[0])
    bayer_matrix -= np.max(bayer_matrix)/2
    # Create an output image with the same dimensions as the input image
    output = np.zeros_like(image)
    # Iterate through the input image
    for y in range(image_height):
        for x in range(image_width):
            pixel_value = image[y, x, :]
            # Find the corresponding threshold value from the Bayer matrix
            threshold = bayer_matrix[y % matrix_size, x % matrix_size]
            # Compare the pixel value to the threshold and map it to the nearest palette color
            output[y, x, :] = findClosest(pixel_value+threshold, palette, type='linear')
    return output


@njit(cache=True)
def closest(image, palette, dist='linear'):
    # Get the dimensions of the input image
    image_height, image_width, _ = image.shape
    # Create an output image with the same dimensions as the input image
    output = np.zeros_like(image)
    # Iterate through the input image
    for y in range(image_height):
        for x in range(image_width):
            pixel_value = image[y, x, :]
            # Compare the pixel value to the threshold and map it to the nearest palette color
            output[y, x, :] = findClosest(pixel_value, palette, type=dist)
    return output


def open_image(filename, gamma=1):
    image= io.imread(filename).astype('float64')
    if np.max(image) > 1:
        image /= 255
    if image.shape[2] == 4:
        image= image[:, :, 0:3]
    image = image**gamma
    return image


def save_image(image, filename):
    image= Image.fromarray((image * 255).astype(np.uint8))
    image.save(filename)


class dimg:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.img = open_image(path)
        self.result = self.img
        self.palette = BW
        self.algorithm = ''

    def resize(self, target_width=None, target_height=None):
        img = np.copy(self.result)
        if target_width and target_height:
            resized_img = resize(img, (target_width, target_height))
        elif target_width:
            width_percent = (target_width / float(img.shape[0]))
            target_height = int((float(img.shape[1]) * float(width_percent)))
            resized_img = resize(img, (target_width, target_height))
        elif target_height:
            height_percent = (target_height / float(img.shape[1]))
            target_width = int((float(img.shape[0]) * float(height_percent)))
            resized_img = resize(img, (target_width, target_height))
        else:
            raise ValueError("Either target_width or target_height must be specified.")
        self.algorithm += '_resized'  
        self.result = resized_img
    
    def square_crop(self):
        self.algorithm += '_square'
        self.result = crop(self.result, squareCropCoordinate(self.result))

    def closest(self, dist = 'euclidean'):
        self.algorithm += '_closest'
        self.result = closest(self.result, self.palette, dist)

    def simple(self):
        self.algorithm += '_simple'
        self.result = dithering_simple(self.result, self.palette)
    
    def fs(self):
        self.algorithm += '_fs'
        self.result = dithering_fs(self.result, self.palette)

    def atk(self):
        self.algorithm += '_atk'
        self.result = dithering_atk(self.result, self.palette)

    def jjn(self):
        self.algorithm += '_jjn'
        self.result = dithering_jjn(self.result, self.palette)
    
    def bayer(self, matrix_size):
        self.algorithm += f'_bayer{matrix_size}'
        self.result = dithering_bayer(self.result, self.palette, matrix_size)
    
    def stucki(self):
        self.algorithm += '_stucki'
        self.result = dithering_Stucki(self.result, self.palette)
    
    def burkes(self):
        self.algorithm += '_burkes'
        self.result = dithering_Burkes(self.result, self.palette)
    
    def save(self):
        output_path = str(self.path)[:-len(self.path.suffix)]+self.algorithm+'.png'
        save_image(self.result, output_path)


def precompile():
    ar= (np.array([[[0, 0, 0], [255, 255, 255]]]) / 255).astype(np.uint8)
    palette= np.array([[[32, 26, 102],
                      [191, 59, 38],
                      [76, 230, 46],
                      [255, 255, 255]]]) / 255
    squareCropCoordinate(ar)
    dithering_atk(ar, palette)
    dithering_jjn(ar, palette)
    dithering_fs(ar, palette)
    dithering_simple(ar, palette)
    dithering_bayer(ar, palette, 4)
    print('precompiling done')