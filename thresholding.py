import os

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

def apply_sauvola_threshold(dir, fn):
    path = os.path.join(dir, fn)
    f, _ = os.path.splitext(fn)
    image = cv.imread(path, 0)
    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    binary_sauvola = image > thresh_sauvola
    cv.imwrite(f'processed/step1_sauvola/{f}.png', binary_sauvola * 255)

def compare_thresholds(path):
    matplotlib.rcParams['font.size'] = 9

    image = cv.imread(path, 0)
    binary_global = image > threshold_otsu(image)

    window_size = 25
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    plt.figure(figsize=(8, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Global Threshold')
    plt.imshow(binary_global, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(binary_niblack, cmap=plt.cm.gray)
    plt.title('Niblack Threshold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    plt.title('Sauvola Threshold')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    dir = 'input_data'
    for path in os.listdir(dir):
        if path.endswith('.png') or path.endswith('.jpg'):
            apply_sauvola_threshold(dir, path)