import cv2 as cv
import numpy as np
# import libs.image_inpainting.main as region_filling
import runpy
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------------------- #
# ---------------------------------------- Masks ---------------------------------------- #
# --------------------------------------------------------------------------------------- #

# Increase/decrease the radius of a binary mask
# @mask: image of mask
# @dilate: amount of pixels to grow or shrink 
def dilate_mask(mask, dilate):
    k = abs(dilate)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    if dilate >= 0:
        return cv.dilate(mask, kernel)
    return cv.erode(mask, kernel)

# Mask using thresholding
# @image_path: path of image
# @dilate: number of pixels to extend mask by
def threshold_mask(image, threshold=115, dilate=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    mask = dilate_mask(mask, dilate)        
    inverse_mask = cv.bitwise_not(mask)
    
    return inverse_mask

def test_mask(image, threshold=0, dilate=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 30, 120)

    # Slightly thicken to make a solid mesh
    kernel = np.ones((3,3), np.uint8)
    mesh = cv.dilate(edges, kernel, 1)
    
    inverse_mesh = cv.bitwise_not(mesh)


    # cv2.imwrite("mesh_edges.png", inverse_mesh)
    return mesh


# --------------------------------------------------------------------------------------- #
# ------------------------------------- Inpainting -------------------------------------- #
# --------------------------------------------------------------------------------------- #
# Inpainting using navier strokes
# @image_path: original image path. mask path is assumed to be in `01_photos_mask`
# @resize: scale image. useful when image is too large and runs too long
def inpaint_navier_strokes(image, mask):
    inpainted_image = cv.inpaint(image, mask, 1, cv.INPAINT_NS)
    return inpainted_image
    

def inpaint_fast_marching_method(image, mask):
    inpainted_image = cv.inpaint(image, mask, 1, cv.INPAINT_TELEA)
    return inpainted_image

def inpaint_region_filling():
    # TODO

    return


def mask_and_inpaint(image, resize=0.3):
    scaled_image = cv.resize(image, None, fx=resize, fy=resize)
    # mask = threshold_mask(scaled_image, 115, -1)
    mask = test_mask(scaled_image, 115, 5)
    inpainted_image = inpaint_navier_strokes(scaled_image, mask)
    
    cv.imwrite(r"..\01_photos_scaled\banana_scaled.png", scaled_image)
    cv.imwrite(r"..\01_photos_mask\banana_mask.png", mask)
    cv.imwrite(r"..\02_photos_inpainted\banana_inpainted.png", inpainted_image)
    return

if __name__ == "__main__":
    image_path = r"..\00_photos_original\soap.png"
    image = cv.imread(image_path)
    mask_and_inpaint(image, 0.3)