import cv2 as cv
import numpy as np
import os


def dilate_mask(mask, dilate):
    k = abs(dilate)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    if dilate >= 0:
        return cv.dilate(mask, kernel)
    return cv.erode(mask, kernel)
    
# @image_path: path of image
# @dilate: number of pixels to extend mask by
def threshold_mask(image_path, threshold=115, dilate=1):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    mask = dilate_mask(mask, -5)        
    inverse_mask = cv.bitwise_not(mask)
    
    cv.imwrite(r"..\01_photos_mask\banana_mask.png", inverse_mask)
    

def grabcut_mask(img):
    # TODO
    return

# threshold_mask(r"..\00_photos_original\banana.png", 115, 5)