import os

import numpy as np
import cv2 as cv
from skimage.filters import threshold_sauvola

from utils import *


def apply_sauvola_threshold(dir, fn):
    image = read_image(dir, fn)
    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    binary_sauvola = (image > thresh_sauvola) * 255
    write_image('step1_sauvola', fn, binary_sauvola)


def find_biggest_contour(dir, fn):
    image = read_image(dir, fn)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(image, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cnts, _ = cv.findContours(binary.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    largest_contour = cnts[0]
    mask = np.zeros_like(image)
    cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    result = cv.bitwise_and(image, mask)
    write_image('step2_biggestcontour', fn, result)


def remove_lines(dir, fn):
    image = read_image(dir, fn)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply adaptiveThreshold at the bitwise_not of gray
    gray = cv.bitwise_not(image)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    # remove horizontal and vertical lines
    mask = cv.bitwise_or(horizontal, vertical)
    result = cv.bitwise_xor(bw, mask)
    # reverse
    result = cv.bitwise_not(result)
    write_image('step3_lines', fn, result)


def processing():
    # dir = 'input_data'
    # for path in os.listdir(dir):
    #     if path.endswith('.png') or path.endswith('.jpg'):
    #         apply_sauvola_threshold(dir, path)
    # dir = 'processed/step1_sauvola'
    # for path in os.listdir(dir):
    #     if path.endswith('.png'):
    #         find_biggest_contour(dir, path)
    dir = 'processed/step2_biggestcontour'
    for path in os.listdir(dir):
        print(path)
        if path.endswith('.png'):
            remove_lines(dir, path)


if __name__ == '__main__':
    processing()