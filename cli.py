#!/usr/bin/env python
import argparse
import simpleDither as sd  # Replace 'dithering_module' with the actual name of your dithering module
from pprint import pformat


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="simpleDither CLI")

    # Required argument: image_path with short and long options
    parser.add_argument("path", help="Path to the input image file")

    # Optional arguments
    parser.add_argument("-wd", "--width", type=int, help="Width for resizing the image (optional)")
    parser.add_argument("-ht", "--height", type=int, help="Height for resizing the image (optional)")
    parser.add_argument("-sc", "--square_crop", action="store_true", help="Perform square crop on the image (optional)")

    algo_choices = [
        "closest", "simple", "fs", "atk", "jjn", "burkes", "stucki", "twoRowSiera", "sierra"
    ]
    parser.add_argument("-a", "--algorithm", default="sierra", help=f"Dithering algorithm (default: sierra). Available:{pformat(algo_choices)}")
    parser.add_argument("-m", "--matrix_size", type=int, default=16, help="In case of bayer dithering, the matrix size (it must be = n^2)")

    color_choices = [
        'BW', 'PICO8', 'RGB', 'APPLE_II', 'AMSTRAD_CPC', 'COMMODORE_64', 'WLK44', 'TWOBIT_DEMICHROME', 'GAMEBOY' 
    ]
    parser.add_argument("-c", "--color_palette", default="BW", help=f"Color palette (default: BW). Available:{pformat(color_choices)}")

    args = parser.parse_args()

    img_path = args.path

    img = sd.dimg(img_path)
    img.palette = sd.colorPalette[args.color_palette]
    if args.width or args.height:
        img.resize(target_width=args.width, target_height=args.height)

    algorithm_name = args.algorithm
    if hasattr(img, algorithm_name):
        algorithm_to_execute = getattr(img, algorithm_name)
        if algorithm_name == 'bayer':
            algorithm_to_execute(args.matrix_size)
        else:
            algorithm_to_execute()
    else:
        print(f"Invalid method: {algorithm_name}")
    
    img.save()

if __name__ == "__main__":
    main()