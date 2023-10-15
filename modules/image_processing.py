from pathlib import Path

import numpy as np
from numba import jit
from PIL import Image
from skimage import io
from skimage.transform import resize
from skimage.util import crop

@jit(nopython=True, cache=True)
def squareCropCoordinate(image):
    """
    The squareCropCoordinate function takes an image as input and returns the coordinates of the largest square possible in that image.
    
    :param image: Pass the image to be cropped
    :return: The coordinate
    :doc-author: Trelent
    """
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


@jit(nopython=True, cache=True)
def findClosest(value, palette):
    """
    The findClosest function takes in a value and a palette.
    The value is an RGB triplet, while the palette is an array of RGB triplets.
    It returns the closest color to that value from the palette.
    
    :param value: Find the closest color in the palette to it
    :param palette: input palette
    :return: The closest color in the palette to a given rgb value
    :doc-author: Trelent
    """
    x = palette[0, :, 0]
    y = palette[0, :, 1]
    z = palette[0, :, 2]
    dx = x - value[0]
    dy = y - value[1]
    dz = z - value[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return palette[0, np.argmin(dist), :]


@jit(nopython=True, cache=True)
def dithering_atk(image, palette):
    """
    The dithering function takes an image and a palette of colors, and returns
    a new image with the same dimensions as the original. The function uses dithering
    to constrain all pixels in the output to be one of the colors in the palette.
    The algorithm used is Atkinson's Dithering Algorithm.
    
    :param image: Pass the image to be processed
    :param palette: Specify the color palette to use
    :param bias: Adjust the error distribution
    :return: A new image that has been dithered
    :doc-author: Trelent
    """
    image = np.concatenate(
        (np.zeros((image.shape[0], 1, 3)), image,
         np.zeros((image.shape[0], 2, 3))),
        axis=1
    )
    image = np.concatenate(
        (image, np.zeros((2, image.shape[1], 3))),
        axis=0
    )
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


@jit(nopython=True, cache=True)
def dithering_jjn(image, palette):
    """
    The dithering function takes an image and a palette of colors, and returns
    a new image with the same dimensions as the original. The function uses dithering
    to constrain all pixels in the output to be one of the colors in the palette.
    The algorithm used is Jarvis-Judice-Ninke Dithering Algorithm.
    
    :param image: Pass the image to be processed
    :param palette: Specify the color palette to use
    :return: A new image that has been dithered
    :doc-author: Trelent
    """
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (np.zeros((image.shape[0], 2, 3)), image,
         np.zeros((image.shape[0], 2, 3)),
         np.zeros((image.shape[0], 1, 3))),
        axis=1
    )
    image = np.concatenate(
        (image, np.zeros((2, image.shape[1], 3))),
        axis=0
    )
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
    output = output[:-2, 2:-2, :]
    
    # Clipping
    output = np.minimum(output, np.ones(output.shape))
    output = np.maximum(output, np.zeros(output.shape))
    
    return output

@jit(nopython=True, cache=True)
def dithering_simple(image, palette):
    """
    The dithering function takes an image and a palette of colors, and returns
    a new image with the same dimensions as the original. The function uses dithering
    to constrain all pixels in the output to be one of the colors in the palette.
    The algorithm used is a simple 2D error diffusion matrix.
    
    :param image: Pass the image to be processed
    :param palette: Specify the color palette to use
    :return: A new image that has been dithered
    :doc-author: Trelent
    """
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (image,
         np.zeros((image.shape[0], 1, 3))),
        axis=1
    )
    image = np.concatenate(
        (image, np.zeros((1, image.shape[1], 3))),
        axis=0
    )
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

@jit(nopython=True, cache=True)
def dithering_fs(image, palette):
    """
    The dithering function takes an image and a palette of colors and returns
    a new image with the same dimensions as the original. The function uses dithering
    to constrain all pixels in the output to be one of the colors in the palette.
    The algorithm used is the Floyd-Steinberg Dithering Algorithm.

    :param image: Pass the image to be processed
    :param palette: Specify the color palette to use
    :return: A new image that has been dithered
    :doc-author: Trelent
    """
    # Extend the image dimensions for error diffusion
    image = np.concatenate(
        (np.zeros((image.shape[0], 1, 3)), image,
         np.zeros((image.shape[0], 1, 3))),
        axis=1
    )
    image = np.concatenate(
        (image, np.zeros((1, image.shape[1], 3))),
        axis=0
    )
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



def open_image(filename):
    """
    The open_image function takes a filename as input and returns an image.
    
    :param filename: Specify the path to the image file
    :return: A numpy array of the image
    :doc-author: Trelent
    """
    image = io.imread(filename).astype('float64')
    if np.max(image) > 1:
        image /= 255
    if image.shape[2] == 4:
        image = image[:, :, 0:3]
    return image

def save_image(image, filename):
    """
    The save_image function takes an image and a filename as input.
    It then saves the image to the specified file name.
    
    :param image: Pass in the image that is to be saved
    :param filename: Name the file that is saved
    :return: Nothing
    :doc-author: Trelent
    """
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(filename)


def processPicture(filename, size, palette, algo='atk'):
    """
    The processPicture function takes in a filename, size, and palette.
    It opens the image from the filename and resizes it to the given size.
    Then it applies dithering to that image using the given palette.
    Finally, it saves this new image with a modified name.

    :param filename: Open the image file
    :param size: Resize the image to a certain size
    :param palette: Determine the colors that will be used in the dithering process
    :return: Nothing
    :doc-author: Trelent
    """
    input_image = open_image(filename)
    image = crop(input_image, squareCropCoordinate(input_image))
    image = resize(image, (size[0], size[1], 3))
    if algo == 'atk': 
        out = dithering_atk(image, palette)
    elif algo == 'jjn':
        out = dithering_jjn(image, palette)
    elif algo == 'fs':
        out = dithering_fs(image, palette)
    elif algo == 'simple':
        out = dithering_simple(image, palette)
    path = Path(filename)
    output_path = f'{path.parent}/{path.stem}_dithered_{algo}{path.suffix}'
    save_image(out, output_path)