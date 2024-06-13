import numpy as np
import os
import cv2 as cv

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def main(path):
    # Load the image
    src = cv.imread(path, cv.IMREAD_COLOR)
    cv.imshow("src", src)
    
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    show_wait_destroy("gray", gray)

    # Apply adaptiveThreshold at the bitwise_not of gray
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    show_wait_destroy("binary", bw)

    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    show_wait_destroy("horizontal", horizontal)
    
    rows = vertical.shape[0]
    verticalsize = rows // 20
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    # Show extracted vertical lines
    show_wait_destroy("vertical", vertical)
    # remove horizontal and vertical lines
    mask = cv.bitwise_or(horizontal, vertical)
    result = cv.bitwise_xor(bw, mask)
    # Show final result
    show_wait_destroy("mask", mask)
    show_wait_destroy("result", result)
    return 0

if __name__ == "__main__":
    dir = 'contours'
    for path in os.listdir(dir):
        main(os.path.join(dir, path))