from collections import defaultdict
import os 
import re

import cv2 as cv 
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from pytesseract import Output

from utils import *

def find_contours(dir, fn, save_image: bool = True):
    image = read_image(dir, fn)
    image = cv.bitwise_not(image)
    cnts, hierarchy = cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # remove biggest contour
    cnts = cnts[1:]
    if save_image:
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        cv.drawContours(image_color, cnts, -1, (0, 255, 0), 3)
        write_image('step4_contours', fn, image_color)
    return image, cnts, hierarchy

def sort_contours(cnts, hierarchy):
    # sort based on hierarchy[3] value
    # start with the parents (lowest number), then the children (highest number)
    # this way we can check the parents first and then the children
    # Enumerate the contours with their original indices
    sort_contours = sorted(enumerate(zip(cnts, hierarchy[0])), key=lambda x: x[1][1][3])  # reverse=True for children first
    index_mapping = {original_index: new_index for new_index, (original_index, _) in enumerate(sort_contours)}
    sorted_cnts, sorted_hierarchy = zip(*[x[1] for x in sort_contours])
    return sorted_cnts, sorted_hierarchy, index_mapping

def extract_contour_and_bounding_box(image, contour, erode_dilate: bool = True):
    mask = np.zeros_like(image)
    cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
    extracted_contour = cv.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv.boundingRect(contour)
    bounding_box = extracted_contour[y:y+h, x:x+w]
    bounding_box = cv.bitwise_not(bounding_box)
    bounding_box = cv.copyMakeBorder(bounding_box, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=(255, 255, 255))
    if erode_dilate:
        bounding_box = apply_dilation_and_erosion(bounding_box)
    return bounding_box

def apply_dilation_and_erosion(image):
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv.dilate(image, kernel, iterations=1)
    erosion = cv.erode(dilation, kernel, iterations=1)
    return erosion


def run_pytesseract(image):
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    data = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
    return data


def run_easyocr(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image, batch_size=1, workers=1, allowlist='0123456789')
    
    data = {'text': [], 'conf': []}
    for result in results:
        text = result[1]
        conf = result[2]
        data['text'].append(text)
        data['conf'].append(conf)
    
    return data


def find_numbers(image, cnts, hierarchy, index_mapping, fn):
    dont_check_these_children = set()
    detected_numbers = []
    for n, (cnt, h) in enumerate(zip(cnts, hierarchy)):
        if n in dont_check_these_children:
            continue
        bounding_box = extract_contour_and_bounding_box(image, cnt)
        data = run_pytesseract(bounding_box)
        for nt, txt in enumerate(data['text']):
            if len(txt) == 0 and data['conf'][nt] < 60:
                continue
            else:
                xmin, ymin = cv.boundingRect(cnt)[:2]
                detected_numbers.append({'text': txt, 'conf': data['conf'][nt], 'contour': cnt, 'xmin': xmin, 'ymin': ymin})
                # add children and children from children to dont_check_these_children
                if h[2] != -1:
                    dont_check_these_children.add(index_mapping[h[2]])
                    while hierarchy[h[2]][1] != -1:
                        h = hierarchy[index_mapping[h[2]]]
                        dont_check_these_children.add(index_mapping[h[1]])
                # write image with bounding box to correct folder
                # if txt is not 0-9 (one digit), write to folder 'rest'
                # if re.match(r'[0-9]', txt) and len(txt) == 1:
                #     write_image(f'../data/train/{str(txt)}', f'{fn}_{str(n)}_{str(txt)}', bounding_box)
                # else:
                #     write_image('../data/train/rest', f'{str(fn)}_{str(n)}_{str(txt)}', bounding_box)
    return detected_numbers


def draw_contours_and_add_text(image, fn, detected_numbers):
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for num in detected_numbers:
        cv.drawContours(image_color, [num['contour']], -1, (0, 255, 0), 3)
        cv.putText(image_color, f"{num['text']}: {num['conf']}", (num['xmin'], num['ymin']), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    write_image('step5_detectednumbers', fn, image_color)


def main():
    dir = 'processed/step2_biggestcontour'
    for path in os.listdir(dir):
        if path.endswith('.png') or path.endswith('.jpg'):
            if path.startswith('IMG_1535'):
                image, cnts, hierarchy = find_contours(dir, path)
                cnts, hierarchy, index_mapping = sort_contours(cnts, hierarchy)
                detected_numbers = find_numbers(image, cnts, hierarchy, index_mapping, path.split('.')[0])
                draw_contours_and_add_text(image, path, detected_numbers)


if __name__ == '__main__':
    main()
    # salt and pepper denoising
    # retina net + CTC (connectionist temporal classification)
    # Check! Row values sum should be equal to column values sum
    # Calculate size (area) per children contours. If it is really small > probably an inner circle; if it’s average, it’s probably right. Show distributions of numbers and sizes.
    # Remove parents of parents OR check parents of 0, 3 and 4 (goes wrong more often). OR detect all and choose the average highest confidence per child/parent.
