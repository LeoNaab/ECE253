# Obstruction Destruction by Byron Chan, Leo Naab, Allen Keng 

## Reflection Obstruction Algorithms:

### Process Images

reflection_obstruction/process_images.py is used by defining the folder path on line 17 for the videos, and then inputting a string
for the processing technique. The process function is called with that string and a folder is created for the output images.

Example calling:
```
process("xue") # in code

python process_images.py # in terminal
```

### Dataset Classifer
reflection_obstruction/dataset_classifier.py has a classify_dataset function that takes in a path to the classification folder and a boolean
for if the folder contains images or videos. 
Example calling:

```
classify_dataset("dataset/unprocessed", video=True) # in code

python dataset_classifier.py # in terminal
```

#### Other files
The other files show examples of other techniques, some not fully implemented, as well as helpers for the main two files. They can be ignored or used as reference for other processing options.

## Opaque Obstruction Algorithms:
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



### How to run
`python ./opaque_obstruction/src/process_images.py` reads all photos from `00_photos_original` and processes the images, which are promptly stored in [Folder structure](#folder-structure). 

### Image Classifier
`python ./opaque_obstruction/src/classifier.py` classifies all the images from the directories `opaque_obstruction/01_photos_scaled` and `opaque_obstruction/02_results_*` and 
outputs results to `classifier_output.txt`


### Folder structure
- `opaque_obstruction/00_photos_original`: Original photos to be processed

---

- `opaque_obstruction/01_photos_mask`: Image's mask using threshold `*_th` and adaptive threshold `*_ad`
- `opaque_obstruction/01_photos_scaled`: Scaled images of the original photos. Images are scaled so that my computer can process the images quicker `*_scaled`
  
---

- `opaque_obstruction/02_results_blurred_gs`: Contains results using gaussian blurring
- `opaque_obstruction/02_results_blurred_med`: Contains results using median blurring
- `opaque_obstruction/02_results_inpainted_ad`: Contains results using adaptive thresholding. `*_fmm` is fast marching method, `*_ns` is navier-strokes.
- `opaque_obstruction/02_results_inpainted_th`: Contains results using simple thresholding. Uses adaptive thresholding. `*_fmm` is fast marching method, `*_ns` is navier-strokes.
  
## Glare Distortion Removal/Reduction Algorithms
Here are 3 techniques we explored and tested on glare distorted images. 
* To try your own images, add your images to the [`glare_distortion/glare_images`](./glare_distortion/glare_images/) directory.
* Ensure that they are in named in the format of `glare{number}{label}.jpg`
  * i.e: `glare1sunglasses.jpg` or  `glare54balloon.jpg`
  * We use the filename to extract the correct label of the object in the image.

### Glare Adaptive Thresholding Traditional Method:
* Referenced from OpenCV <sup>1</sup>
  * It determines the threshold for a pixel based on the small area surrounding it
  * Using "threshold value that is a gaussian-weighted sum of the neighbourhood values minus the constant C." 
  * Finetuned the block size to see relatively well-outlined objects for most images.

* Open the Jupyter Notebook [`compareAdaptiveThresholding.ipynb`](./glare_distortion/compareAdaptiveThresholding.ipynb)
* Run all.
  * The black and white images will be saved to the `glare_thresholding_inputs` directory.
  * The thresholded images will be saved to the `glare_thresholding_outputs` directory. 
  * The code compares these two images as a fair difference. 
    * (The object classifier will obviously do better with the original colored image.)
  * The bottom will show how many gray images and processed images got correct.
  * It will also show the probabilities in which the processed image improved.

### Glare Reduction Traditional Method:
* Pulled from Tran et al.<sup>2</sup> 
  * Reduce-glare filter via polynomial using intensity thresholding.
  * Enhance contract: f = 1.6
  * Reduce-glare filter via polynomial using intensity thresholding.
  * Enhance contract: f = 1.4

* Put glare distorted images in the `glare_images` directory. 
* Run the commands
```
$ cd glare-reduction-existing/
$ python generate.py
$ cd ..
```
* The resulting images will be saved to the `glare_reduction_outputs` directory.
* To compare the before and after images with the ResNet50_Weights.IMAGENET1K_V1 pretrained model. 
  * Open the Jupyter Notebook [`compareGlareReduction.ipynb`](./glare_distortion/compareGlareReduction.ipynb)
  * Run all. 

### Deep Trident Decomposition Network for Single License Plate Image Glare Removal
* Reference from Chen et al.<sup>3</sup>  
  * Used their pretrained model for removing glare from license plates.
  * Had to resort to resolution of (256, 192) unfortunately, so a lot of interpolation.
  * Designed to clean up intense glare from images.

* [***NOTE***]: We had to use Python 3.6.0 to run their code. 
  * Other dependencies can be found in [`requirement.txt`](./glare_distortion/glare-removal-dnn-pretrained/requirement.txt) 
  * We set up a python virtual environment using (in Windows Powershell) and ran as such
  ```
  $ virtualenv venv --python="{filepath to Python3.6}" # one time
  $ .\env\Scripts\Activate.ps1
  ...
  python test.py --test_path=../glare_images/ --load_pretrain=./save_weight/model.h5
  ...
  $ deactivate
  ```
  * It should output inside the repository in the [`test_result`](./glare_distortion/glare-removal-dnn-pretrained/test_result/) directory.
  * Open the Jupyter Notebook [`compareGlareRemovalPretrained.ipynb`](./glare_distortion/compareGlareRemovalPretrained.ipynb)
  * Run all. 

# Citations:
[1]: OpenCV. https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html   
[2]: Tran, D. T. (2019). Glare Reduction [Computer software]. https://github.com/ducthotran2010/glare-reduction  
[3]: Chen et al. (2022). Deep Trident Decomposition Network for Single License Plate Image Glare Removal. https://github.com/bigmms/chen_tits21
