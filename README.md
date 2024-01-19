# simpleDither
## what is it ?
a simple and exhaustive dithering program that feature a lot of different algorithm and color palette!

## how do I use it ?
```python
import simpleDither as sd

img_path = 'path/to/image.JPG'

# create a dimg instance
img = sd.dimg(img_path)

# select a palette (default is black and white)
img.palette = sd.GAMEBOY

# resize to see those juicy pixels
img.resize(target_width=160)

# apply the algorithm of your choice
img.bayer(16)

# save the result !
img.save()
```
