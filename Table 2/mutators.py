from __future__ import print_function
from importlib import reload
import sys
import cv2
import numpy as np
import random
import time
import copy
reload(sys)
# sys.setdefaultencoding('utf8')



# keras 1.2.2 tf:1.2.0
class Mutators():
    def image_translation(img, params):

        rows, cols, ch = img.shape
        # rows, cols = img.shape

        # M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        M = np.float32([[1, 0, params], [0, 1, params]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_scale(img, params):

        # res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
        rows, cols, ch = img.shape
        res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
        res = res.reshape((res.shape[0],res.shape[1],ch))
        y, x, z = res.shape
        if params > 1:  # need to crop
            startx = x // 2 - cols // 2
            starty = y // 2 - rows // 2
            return res[starty:starty + rows, startx:startx + cols]
        elif params < 1:  # need to pad
            sty = (rows - y) // 2
            stx = (cols - x) // 2
            return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant', constant_values=0)
        return res

    def image_shear(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        factor = params * (-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_rotation(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
        return dst

    def image_contrast(img, params):
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
        # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

        return new_img

    def image_brightness(img, params):
        beta = params
        new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
        return new_img

    def image_blur(img, params):

        # print("blur")
        blur = []
        if params == 1:
            blur = cv2.blur(img, (3, 3))
        if params == 2:
            blur = cv2.blur(img, (4, 4))
        if params == 3:
            blur = cv2.blur(img, (5, 5))
        if params == 4:
            blur = cv2.GaussianBlur(img, (3, 3), 0)
        if params == 5:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        if params == 6:
            blur = cv2.GaussianBlur(img, (7, 7), 0)
        if params == 7:
            blur = cv2.medianBlur(img, 3)
        if params == 8:
            blur = cv2.medianBlur(img, 5)
        # if params == 9:
        #     blur = cv2.blur(img, (6, 6))
        if params == 9:
            blur = cv2.bilateralFilter(img, 6, 50, 50)
            # blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur

    def image_pixel_change(img, params):
        # random change 1 - 5 pixels from 0 -255
        img_shape = img.shape
        img1d = np.ravel(img)
        arr = np.random.randint(0, len(img1d), params)
        for i in arr:
            img1d[i] = np.random.randint(0, 256)
        new_img = img1d.reshape(img_shape)
        return new_img

    def image_noise(img, params):
        if params == 1:  # Gaussian-distributed additive noise.
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            return noisy.astype(np.uint8)
        elif params == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in img.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in img.shape]
            out[tuple(coords)] = 0
            return out
        elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            row, col, ch = img.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = img + img * gauss
            return noisy.astype(np.uint8)
















