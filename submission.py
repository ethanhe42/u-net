from __future__ import print_function

import numpy as np
import cv2
from data import image_cols, image_rows
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
from PIL import Image, ImageOps, ImageEnhance

from skimage.transform import rotate, resize
from skimage import data

def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def augmentation(image, org_width=160,org_height=224, width=190, height=262):
    max_angle=20
    image=resize(image,(width,height))

    angle=np.random.randint(max_angle)
    if np.random.randint(2):
        angle=-angle
    image=rotate(image,angle,resize=True)

    xstart=np.random.randint(width-org_width)
    ystart=np.random.randint(height-org_height)
    image=image[xstart:xstart+org_width,ystart:ystart+org_height]

    if np.random.randint(2):
        image=cv2.flip(image,1)
    
    if np.random.randint(2):
        image=cv2.flip(image,0)
    # image=resize(image,(org_width,org_height))

    print(image.shape)
    plt.imshow(image)
    plt.show()

def visualize():
    from data import load_train_data
    imgs_train, imgs_train_mask = load_train_data()
    imgs_train_pred=np.load('imgs_train_pred.npy')
    total=imgs_train.shape[0]
    for i in range(total):
        # augmentation(imgs_train[i,0])
        plt.subplot(221)
        plt.imshow(imgs_train[i,0])
        plt.subplot(222)
        plt.imshow(imgs_train_mask[i,0])
        plt.subplot(223)
        plt.imshow(imgs_train_pred[i,0])
        img = prep(imgs_train_pred[i,0])
        plt.subplot(224)
        plt.imshow(img)
        plt.show()

def submission():
    from data import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_mask_test.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    # submission()
    visualize()
    # while True:
    #     augmentation()
