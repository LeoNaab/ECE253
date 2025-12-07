import cv2 as cv
import numpy as np
from skimage.restoration import wiener
from pathlib import Path



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

# Mask using adaptive thresholding
def adaptive_mask(image, threshold=15, dilate=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    mesh_mask = cv.adaptiveThreshold(
        blurred, 
        255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 
        threshold, # adjustable
        2 # adjustable
    )
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = dilate_mask(mesh_mask, dilate)
    final_mask = cv.erode(dilated_mask, kernel, iterations=1)

    return final_mask

# Mask using canny edge detection. Does not work
def canny_mask(image, threshold=0, dilate=0):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    low_threshold = 200
    high_threshold = 2*low_threshold
    edges = cv.Canny(blurred, low_threshold, high_threshold)
    mask = np.zeros_like(edges)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(mask, contours, -1, 255, -1)
    
    mask = cv.bitwise_not(mask)
    return mask

# It is difficult to get grabcut to select the foreground.
# The reasoning behind this is because the goal of GrabCut is to "select"
# an object within a bounded rectangle. However, the obstruction, (microwave mesh) is
# not an object in the image, and rather, part of the whole image and on the foreground
def grabcut_mask(image, threshold=0, dilate=0):
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (500,500,1000,1000)
    cv.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # img = image*mask2[:,:,np.newaxis]
    
    final_mask = np.where(
        (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),
        255, 0
    ).astype('uint8')
    
    return final_mask

# --------------------------------------------------------------------------------------- #
# ------------------------------------- Inpainting -------------------------------------- #
# --------------------------------------------------------------------------------------- #
# Inpainting using navier strokes
# @image_path: original image path. mask path is assumed to be in `01_photos_mask`
# @resize: scale image. useful when image is too large and runs too long
def inpaint_navier_strokes(image, mask, radius=1):
    inpainted_image = cv.inpaint(image, mask, radius, cv.INPAINT_NS)
    return inpainted_image
    

def inpaint_fast_marching_method(image, mask, radius=1):
    inpainted_image = cv.inpaint(image, mask, radius, cv.INPAINT_TELEA)
    return inpainted_image

def inpaint_region_filling():
    # TODO
    return

# --------------------------------------------------------------------------------------- #
# -------------------------------------- Filtering -------------------------------------- #
# --------------------------------------------------------------------------------------- #

def unblur(image, radius=15):    
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    kernel_size = radius
    sigma = radius / 6
    g = cv.getGaussianKernel(kernel_size, sigma)
    kernel = g @ g.T

    restored = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[2]):
        restored[:, :, c] = wiener(img[:, :, c], kernel, balance=0.1)

    restored = np.clip(restored * 255, 0, 255).astype(np.uint8)

    return restored

# Gaussian blur works suprisingly well, but it becomes very dim
# And image is now blurred
def gaussian_blur(image, radius=15):
    return cv.GaussianBlur(image, (radius, radius), 0)

# Median blur does not work well with obstructions. 
# Nonlinear & adds some distortion
def median_blur(image, radius=15):
    return cv.medianBlur(image, radius)


def mask_and_inpaint(image):
    # mask = threshold_mask(image, 115, -1)
    # mask = adaptive_mask(image, 15, 2)
    # mask = canny_mask(image, 0, 0)
    
    th_mask = threshold_mask(image, 115, -1)
    ad_mask = adaptive_mask(image, 15, 2)
    
    inpainted_image_ns_th = inpaint_navier_strokes(image, th_mask, radius=4)
    inpainted_image_fmm_th = inpaint_fast_marching_method(image, th_mask, radius=4)
    
    inpainted_image_ns_ad = inpaint_navier_strokes(image, ad_mask, radius=4)
    inpainted_image_fmm_ad = inpaint_fast_marching_method(image, ad_mask, radius=4)

    return th_mask, ad_mask, inpainted_image_ns_th, inpainted_image_fmm_th, inpainted_image_ns_ad, inpainted_image_fmm_ad

def run_methods(image, file_name, resize=0.3):    
    scaled_image = cv.resize(image, None, fx=resize, fy=resize)

    # Inpaints 
    # Thresholding
    threshold_mask, adaptive_mask, inpainted_image_ns_th, inpainted_image_fmm_th, inpainted_image_ns_ad, inpainted_image_fmm_ad = mask_and_inpaint(scaled_image)
    cv.imwrite(rf"..\01_photos_scaled\{file_name}_scaled.png", scaled_image)
    cv.imwrite(rf"..\01_photos_mask\{file_name}_mask_threshold.png", threshold_mask)
    cv.imwrite(rf"..\01_photos_mask\{file_name}_mask_adaptive.png", adaptive_mask)
    
    cv.imwrite(rf"..\02_results_th\{file_name}_inpainted_ns.png", inpainted_image_ns_th)
    cv.imwrite(rf"..\02_results_th\{file_name}_inpainted_fmm.png", inpainted_image_fmm_th)
    
    cv.imwrite(rf"..\02_results_ad\{file_name}_inpainted_ns.png", inpainted_image_ns_ad)
    cv.imwrite(rf"..\02_results_ad\{file_name}_inpainted_fmm.png", inpainted_image_fmm_ad)

    # Blurs
    blurred_image_gaussian = gaussian_blur(scaled_image, radius=35)
    cv.imwrite(rf"..\02_results_blurred_gs\{file_name}_blur_gaussian.png", blurred_image_gaussian)
    
    blurred_image_median = median_blur(scaled_image, radius=35)
    cv.imwrite(rf"..\02_results_blurred_med\{file_name}_blur_med.png", blurred_image_median)



if __name__ == "__main__":
    file_name = "fountain_pen_1"
    
    image_path = rf"..\00_photos_original\{file_name}.png"    
    image = cv.imread(image_path)
    run_methods(image, file_name)