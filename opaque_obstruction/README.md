# Opaque Obstructions

## Overview
This section is about removing opaque obstructions. There are images of objects behind meshes / microwave screen. There are two main techniques used to remove the obstructions: blurring the whole image, and inpainting.

### Blurring
- Gaussian blur: `gaussian_blur()`
  - \+ Simple
  - \- Obstruction changes the color of the whole image (i.e. microwave makes images dark)

- Median blur: `median_blur()`
  - \- Usually does not work. When it does work, it adds a grid to the foreground
  - \- When it doesn't work, it makes the image very dark

### Inpainting
#### Masking techniques
- Simple thresholding: `threshold_mask()`
  - \+ Removes meshes
  - \+ Small artifacts could be blurred out
  - \- Parts of objects that are the same color as the obstruction are blurred
  - 
- Adaptive thresholding: `adaptive_mask()`
  - \+ Works the best
  - \+ Small artifacts could be blurred out
  - \- Slightly blurry
  - \- Artifacts are more discernable than simple thresholding

- [UNUSED] Canny edge detection: `canny_mask()`
  - \- Does not work. It is able to capture edges, but the lines it generates are too thin and inconsistent to be effective.

- [UNUSED] GrabCut: `grabcut_mask()`
  - \- Does not work. This is because you need to address an explicit estimated area (rectangle) where the selected object/obstruction is located. Because we want to select a foreground obstruction (mesh), and it is on the whole image, it has a difficult time selecting it.

#### Inpainting techniques
- Fast marching method `inpaint_fast_marching_method()`
  - Blurrier image
  - \+ Artifacts are less obvious

- Navier strokes: `inpaint_navier_strokes()`
  - Sharper image
  - \- Artifacts are more obvious

- [UNIMPLEMENTED] Region Filling: `inpaint_region_filling()`
  - \- GitHub library. Does not work on whole images. Also takes a long time to run per image.

#### Improving inpainting techniques
To reduce or remove the artifacts on the post-processed inpainting techniques, they could be blurred using Gaussian blur or median blur, but it would make the image blurrier.
- Median blur is decently effective in removing artifacts

Tuning the masks by adjusting thresholds and dilations per-image could also be effective in removing artifacts.

Increasing the dilation size is able to effectively artifacts. However, it reduces the image quality because more parts of the image is inpainted over.

- All methods to reduce artifacts have the side effect of making the image more blurrier.



## How to run
`python ./src/process_images.py` reads all photos from `00_photos_original` and processes the images, which are promptly stored in [Folder structure](#folder-structure). 

## Image Classifier
`python ./src/classifier.py` classifies all the images from the directories `01_photos_scaled` and `02_results_*` and 
outputs results to `classifier_output.txt`


## Folder structure
- `00_photos_original`: Original photos to be processed

---

- `01_photos_mask`: Image's mask using threshold `*_th` and adaptive threshold `*_ad`
- `01_photos_scaled`: Scaled images of the original photos. Images are scaled so that my computer can process the images quicker `*_scaled`
  
---

- `02_results_blurred_gs`: Contains results using gaussian blurring
- `02_results_blurred_med`: Contains results using median blurring
- `02_results_inpainted_ad`: Contains results using adaptive thresholding. `*_fmm` is fast marching method, `*_ns` is navier-strokes.
- `02_results_inpainted_th`: Contains results using simple thresholding. Uses adaptive thresholding. `*_fmm` is fast marching method, `*_ns` is navier-strokes.

