## Using the python scripts:

### Process Images
process_images.py is used by defining the folder path on line 17 for the videos, and then inputting a string
for the processing technique. The process function is called with that string and a folder is created for the output images.

Example calling:
```
process("xue") # in code

python process_images.py # in terminal
```

### Dataset Classifer
Dataset_classifier.py has a classify_dataset function that takes in a path to the classification folder and a boolean
for if the folder contains images or videos. 
Example calling:

```
classify_dataset("dataset/unprocessed", video=True) # in code

python dataset_classifier.py # in terminal
```

#### Other files
The other files show examples of other techniques, some not fully implemented, as well as helpers for the main two files. They can be ignored or used as reference for other processing options.
