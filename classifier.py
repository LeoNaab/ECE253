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

image_file =  "image_results/post_vulture.png"#"outputs/bananaModified.png"#
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
print(f"Using Weights: {weights}")
print(f"Predicted Label: **{predicted_label}**")
print(f"Probability: {probabilities[predicted_index].item():.4f}")
print("-" * 45)
