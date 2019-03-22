import numpy as np
import imageio
import scipy.misc
from copy import copy

"""Image pool"""
class ImagePool(object):
    def __init__(self, max_pool_size=50):
        self.max_pool_size = max_pool_size
        self.pool_size = 0
        self.images = []

    def __call__(self, image):
        if self.max_pool_size <= 0:
            return image
        if self.pool_size < self.max_pool_size:
            self.pool_size += 1
            self.images.append(image)
            return image
        if np.random.random() < 0.5:
            index = int(np.random.random() * self.max_pool_size)
            im1 = copy(self.images[index])[0]
            self.images[index][0] = image[0]
            index = int(np.random.random() * self.max_pool_size)
            im2 = copy(self.images[index])[1]
            self.images[index][1] = image[1]
            return [im1, im2]

        return image

"""Read image"""
def imread(image_path):
    im = imageio.imread(image_path)
    return im

"""Save image"""
def imsave(image_path, im):
    imageio.imwrite(image_path, im)

"""Resize image"""
def imresize(im, im_size):
    return scipy.misc.imresize(im, [im_size, im_size])

"""Normalize image"""
def imnormalize(im):
    return im / 255

"""Load training data"""
def load_train_data(path_to_images, im_size):
    images = []
    for path in path_to_images:
        im = imread(path)
        im = imresize(im, im_size)

        if np.random.random() < 0.5:
            im = np.fliplr(im)

        imnormalize(im)
        images.append(im)

    return np.asarray(images).astype(np.float32)
