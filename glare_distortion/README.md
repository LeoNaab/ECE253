# Glare Distortion Removal/Reduction Algorithms

### Glare Adaptive Thresholding Traditional Method:
* Referenced from [Citation 1]  
  * It determines the threshold for a pixel based on the small area surrounding it
  * Using "threshold value that is a gaussian-weighted sum of the neighbourhood values minus the constant C." 
  * Finetuned the block size to see relatively well-outlined objects for most images.

* Open the Jupyter Notebook [`compareAdaptiveThresholding.ipynb`](./compareAdaptiveThresholding.ipynb)
* Run all.
  * The black and white images will be in the `glare_thresholding_inputs` directory.
  * The thresholded images will be in the `glare_thresholding_outputs` directory. 
  * The code compares these two images as a fair difference. 
    * (The object classifier will obviously do better with the original colored image.)
  * The bottom will show how many gray images and processed images got correct.
  * It will also show the probabilities in which the processed image improved.

### Glare Reduction Traditional Method:
* Pulled from [Citation 2] 
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
* The resulting images will be in the `glare_reduction_outputs` directory.
* To compare the before and after images with the ResNet50_Weights.IMAGENET1K_V1 pretrained model. 
  * Open the Jupyter Notebook [`compareGlareReduction.ipynb`](./compareGlareReduction.ipynb)
  * Run all. 

### Deep Trident Decomposition Network for Single License Plate Image Glare Removal
* Reference from [Citation 3]  
  * Used their pretrained model for removing glare from license plates.
  * Had to resort to resolution of (256, 192) unfortunately, so a lot of interpolation.
  * Designed to clean up intense glare from images.

* [***NOTE***]: We had to use Python 3.6.0 to run their code. 
  * Other dependencies can be found in [`requirement.txt`](./glare-removal-dnn-pretrained/requirement.txt) 
  * We set up a python virtual environment using (in Windows Powershell) and ran as such
  ```
  $ virtualenv venv --python="{filepath to Python3.6}" # one time
  $ .\env\Scripts\Activate.ps1
  ...
  python test.py --test_path=../glare_images/ --load_pretrain=./save_weight/model.h5
  ...
  $ deactivate
  ```
  * It should output inside the repository in the [`test_result`](./glare-removal-dnn-pretrained/test_result/) directory.
  * Open the Jupyter Notebook [`compareGlareRemovalPretrained.ipynb`](./compareGlareRemovalPretrained.ipynb)
  * Run all. 

# Citations:
[1]: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html 
[2]: Tran, D. T. (2019). Glare Reduction [Computer software]. https://github.com/ducthotran2010/glare-reduction
[3]: Chen et al. (2022). Deep Trident Decomposition Network for Single License Plate Image Glare Removal. https://github.com/bigmms/chen_tits21