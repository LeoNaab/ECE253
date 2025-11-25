# %%
import torch
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np 
import torch.nn.functional as F 
from glob import glob 

# Load stuff for object classification
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval() 
preprocess = weights.transforms()
categories = weights.meta["categories"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_cv_image(cv_img, preprocess_func):
    if cv_img.ndim == 2:
        # Stack 1-channel grayscale to 3 channels for RGB-trained models
        rgb_img = np.stack([cv_img, cv_img, cv_img], axis=2)
    else:
        rgb_img = cv_img

    pil_image = Image.fromarray(rgb_img)
    input_tensor = preprocess_func(pil_image)
    return input_tensor

GLARE_FILES = glob('glare_images/*glare*.jpg')

if not GLARE_FILES:
    print("No files containing 'glare' found in the 'glare_images/' folder.")
else:
    print(f"Found {len(GLARE_FILES)} files to process.")

# Open the markdown file for writing
with open('adaptive_thresholding.md', 'w') as md_file:
    
    for filename in GLARE_FILES:
        md_file.write("\n" + "=" * 30 + "\n")
        
        try:
            object_part_with_num = filename.split('glare')[1]
            
            start_index = 0
            for char in object_part_with_num:
                if char.isdigit():
                    start_index += 1
                else:
                    break
            object_part = object_part_with_num[start_index:] 
            expected_obj = object_part.split('.')[0] 
            
            md_file.write(f"Processing File: {filename}\n")
            md_file.write(f"Expected Object: **{expected_obj.upper()}**\n")
        except IndexError:
            md_file.write(f"Error parsing filename: {filename}. Skipping.\n")
            continue 
        
        image = cv2.imread(filename)
        if image is None:
            md_file.write(f"ERROR: Could not read image {filename}. Skipping.\n")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Adaptive Thresholding
        thresh_gauss = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            121, 3
        )

        # Convert to Tensors for prediction
        input_tensor_gray = preprocess_cv_image(gray_image, preprocess)
        input_tensor_thresh = preprocess_cv_image(thresh_gauss, preprocess)

        
        input_batch = input_tensor_gray.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)

        probabilities = F.softmax(output[0], dim=0)
        
        # Get top 1 result
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = categories[predicted_index]

        md_file.write(f"Top Predicted Label: **{predicted_label}**\n")
        md_file.write(f"Probability: {probabilities[predicted_index].item():.4f}")
        
        md_file.write("\nTop 5 Predicted Categories:\n")
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        for i in range(top5_prob.size(0)):
            predicted_index = top5_indices[i].item()
            predicted_label = categories[predicted_index]
            probability = top5_prob[i].item()
            md_file.write(f"   {i+1}. **{predicted_label}** (Probability: {probability:.4f})\n")
        md_file.write("=" * 30 + "\n")
        md_file.write(f"After Adaptive Thresholding: \n")
        input_batch = input_tensor_thresh.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)

        probabilities = F.softmax(output[0], dim=0)
        # Get top 1 result
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = categories[predicted_index]

        md_file.write(f"Top Predicted Label: **{predicted_label}**\n")
        md_file.write(f"Probability: {probabilities[predicted_index].item():.4f}")

        md_file.write("\nTop 5 Predicted Categories:\n")
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        for i in range(top5_prob.size(0)):
            predicted_index = top5_indices[i].item()
            predicted_label = categories[predicted_index]
            probability = top5_prob[i].item()
            md_file.write(f"   {i+1}. **{predicted_label}** (Probability: {probability:.4f})\n")
        md_file.write("=" * 30 + "\n")

print("Processing complete. Results written to adaptive_thresholding.md")