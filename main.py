#!/usr/bin/env python
import simpleDither as sd

img_path = 'IN/DSCF9150.JPG'

img = sd.dimg(img_path)
img.palette = sd.GAMEBOY
img.resize(target_width=160)
img.bayer(16)
img.save()