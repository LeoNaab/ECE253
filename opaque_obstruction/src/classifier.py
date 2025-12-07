import torch
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from torchvision.transforms import transforms
from PIL import Image
import os

weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

def image_loader(image_name):
    if not os.path.exists(image_name):
        print(f"ERROR: File not found at path: {image_name}")
        return None

    image = Image.open(image_name).convert("RGB")
    return image

def classify_image(image_file):
    image = image_loader(image_file)
    input_tensor = preprocess(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_tensor.unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    predicted_index = torch.argmax(probabilities).item()

    categories = weights.meta["categories"]
    predicted_label = categories[predicted_index]

    print("-" * 45)
    print(f"Predictiong: file {image_file}")
    print(f"Using Weights: {weights}")
    print(f"Predicted Label: **{predicted_label}**")
    print(f"Probability: {probabilities[predicted_index].item():.4f}")
    print("-" * 45)
    
    return predicted_label, probabilities, predicted_index, weights

image_directories = [
    r"..\01_photos_scaled",
    r"..\02_results_blurred_gs",
    r"..\02_results_blurred_med",
    r"..\02_results_inpainted_ad",
    r"..\02_results_inpainted_th"
]

# for folder in image_directories:
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)

#         if os.path.isfile(file_path):
#             classify_image(file_path)
results = []

def extract_sort_key(file_path):
    base_name = os.path.basename(file_path)
    if base_name.endswith("_scaled.png"):
        base_name = base_name[:-11]
    return base_name

for folder in image_directories:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            predicted_label, probabilities, predicted_index, weights = classify_image(file_path)
            
            results.append((
                file_path,
                predicted_label,
                probabilities[predicted_index].item(),
                weights
            ))
results.sort(key=lambda x: extract_sort_key(x[0]))
output_path = "classifier_output.txt"

with open(output_path, "w") as f:
    for file_path, predicted_label, probability, weights in results:
        f.write("-" * 45 + "\n")
        f.write(f"Predicting File: {file_path}\n")
        f.write(f"Using Weights: {weights}\n")
        f.write(f"Predicted Label: **{predicted_label}**\n")
        f.write(f"Probability: {probability:.4f}\n")
        f.write("-" * 45 + "\n\n")