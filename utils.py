import numpy as np
import imageio

"""Read image"""
def imread(image_path):
    im = imageio.imread(image_path)
    return im

"""Save image"""
def imsave(image_path, im):
    imageio.imwrite(image_path, im)
