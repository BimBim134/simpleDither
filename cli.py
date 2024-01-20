#!/usr/bin/env python
import argparse
import simpleDither as sd  # Replace 'dithering_module' with the actual name of your dithering module

algo_choices = [
    "closest", "simple", "fs", "atk", "jjn", "burkes", "stucki", "twoRowSiera", "sierra"
]

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="simpleDither CLI")

    # Required argument: image_path with short and long options
    parser.add_argument("path", help="Path to the input image file")

    # Optional arguments
    parser.add_argument("-wd", "--width", type=int, help="Width for resizing the image (optional)")
    parser.add_argument("-ht", "--height", type=int, help="Height for resizing the image (optional)")
    parser.add_argument("-sc", "--square_crop", action="store_true", help="Perform square crop on the image (optional)")


    parser.add_argument("-a", "--algorithm", default="Sierra", choices=algo_choices, help="Dithering algorithm (default: sierra)")
    parser.add_argument("-m", "--matrix_size", default=16, choices=algo_choices, help="in case of bayer dithering, the matrix size. (it must be = 2^n)")

    color_choices = [
        'BW', 'PICO8', 'RGB', 'APPLE_II', 'AMSTRAD_CPC', 'COMMODORE_64', 'WLK44', 'TWOBIT_DEMICHROME', 'GAMEBOY' 
    ]
    parser.add_argument("-c", "--color_palette", default="BW", choices=color_choices, help="Color palette (default: BW)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the dithering function with the specified parameters
    # dither_image(image_path=args.image_path, width=args.width, height=args.height, square_crop=args.square_crop,
    #              algorithm=args.algorithm, color_palette=args.color_palette)

    print(args)
    


if __name__ == "__main__":
    main()