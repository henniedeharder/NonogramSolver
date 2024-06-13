import os 

import cv2 as cv

def read_image(dir, fn):
    path = os.path.join(dir, fn)
    image = cv.imread(path, 0)
    return image


def write_image(dir, fn, image):
    f, _ = os.path.splitext(fn)
    if not os.path.exists(f'processed/{dir}'):
        os.makedirs(f'processed/{dir}')
    cv.imwrite(f'processed/{dir}/{f}.png', image)


def display_image(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)