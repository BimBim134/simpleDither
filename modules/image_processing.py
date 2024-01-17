from pathlib import Path
from pprint import pprint

import numpy as np
from numba import njit
from PIL import Image
from skimage import io
from skimage.transform import resize
from skimage.util import crop
import matplotlib.pyplot as plt


class Dimg:
    def __init__(self) -> None:
        pass

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


def processPicture(filename, size, palette, algo='atk', matrix_size=4, gamma=1):
    input_image= open_image(filename, gamma)
    image= crop(input_image, squareCropCoordinate(input_image))
    image= resize(image, (size[0], size[1], 3))
    if algo == 'atk':
        out= dithering_atk(image, palette)
    elif algo == 'jjn':
        out= dithering_jjn(image, palette)
    elif algo == 'fs':
        out= dithering_fs(image, palette)
    elif algo == 'bayer':
        out= dithering_bayer(image, palette, matrix_size)
        algo += f'_m{matrix_size}'
    elif algo == 'simple':
        out= dithering_simple(image, palette)
    else:
        out= closest(image, palette)
        algo = 'closest'
    path= Path(filename)
    output_path= f'{path.parent}/{path.stem}_dithered_{algo}_{matrix_size}_{gamma:.1f}.png'
    save_image(out, output_path)
    print(f'saved : {output_path}')


def processPicture_paralel(args):
    filename, size, palette, algo, matrix_size, gamma = args[0], args[1], args[2], args[3], args[4], args[5]
    processPicture(filename, size, palette, algo, matrix_size, gamma)


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


def save_color_palette(palette, filename):
    plt.figure()
    plt.imshow(palette)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
