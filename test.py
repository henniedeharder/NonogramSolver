import os

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# step 2: detect numbers in biggest contour
fn = 'nonogram'
ext = '.png'
image = cv2.imread(f'input_data/{fn}{ext}')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f'data/{fn}_gray.png', gray)
# reverse colors
gray = cv2.bitwise_not(gray)
cv2.imwrite(f'data/{fn}_gray_reverse.png', gray)
# binary threshold
thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite(f'data/{fn}_thresh.png', thresh)
# detect contours
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# only take inner contours :( doesn't work for 6, 8, 9, 0 (4)
# cnts = [c for c, h in zip(cnts, hierarchy[0]) if h[2] == -1]

# draw contours
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
cv2.imwrite(f'data/{fn}_contours.png', image)

# extract bounding boxes around contours and run tesseract OCR on each of them
detected_numbers = []
xmin = 100000
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # only accept contours that have an area between 1 and 100 pixels
    # extract the digit and threshold it to make the digit
    # appear as *white* (foreground) on a *black
    # background, then grab the width and height of the
    # thresholded image
    roi = gray[y:y + h, x:x + w]
    thresh = cv2.threshold(roi, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (tH, tW) = thresh.shape
	# add some white pixels around the image
    thresh = cv2.copyMakeBorder(thresh, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # run tesseract OCR on the thresholded image
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    # loop over each of the individual text localizations
    for i in range(0, len(data["text"])):
        # extract the OCR text itself along with the confidence of the text localization
        text = data["text"][i]
        conf = int(data["conf"][i])
        # filter out
        if conf > 50 and len(text) > 0:
            # extract the bounding box coordinates of the text region from the current result on the original image
            xt = x + data["left"][i]
            yt = y + data["top"][i]
            if xt < xmin:
                xmin = xt
                y_values = (yt, yt+h)
            wt = data["width"][i]
            ht = data["height"][i]
            detected_numbers.append({'text': text, 'xmin': xt, 'xmax': xt+wt, 'ymin': yt, 'ymax': yt+ht, 'w': wt, 'h': ht, 'conf': conf})

print(detected_numbers)
print(xmin)

# distinction between row and column values
# find row where xmin is minimal
# find xmax of all numbers in that row (with same y)
# all numbers with xmin < xmax are row numbers
# all numbers with xmin > xmax are column numbers
values_xmin = [i for i in detected_numbers if y_values[0] < i['ymin'] + .5 * i['h'] < y_values[1]]
xmax = max([i['xmax'] for i in values_xmin])
row_numbers = [i for i in detected_numbers if i['xmin'] < xmax + 5]
column_numbers = [i for i in detected_numbers if i['xmin'] > xmax + 5]
print([i['text'] for i in row_numbers])
print([i['text'] for i in column_numbers])