import os

import imageio

# import cv2
import torch
from PIL import Image
from torch.nn.parallel.replicate import T
from torchvision.models import ResNet50_Weights, resnet50

# from torchvision.transforms import transforms

weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()


def image_loader(image_name, video=True):
    if not os.path.exists(image_name):
        print(f"ERROR: File not found at path: {image_name}")
        return None

    # if video then take middle frame from video using imageio
    if video:
        # print(image_name)
        reader = imageio.get_reader(image_name, "ffmpeg")
        frame = reader.get_data(int(reader.get_meta_data()["duration"] / 2))
        image = Image.fromarray(frame)
    else:
        image = Image.open(image_name).convert("RGB")
    return image


def classify_dataset(path, video=False):
    accurate = 0
    total = 0
    total_prob = 0
    for image_file in os.listdir(path):
        if image_file.startswith("."):  # frickin .DS_store
            continue

        ground_truth = image_file.split(".")[0].replace("_", " ")

        # print(ground_truth)

        # image_file = "image_results/pre_vulture.png"  # "outputs/bananaModified.png"#
        image = image_loader(f"{path}/{image_file}", video)

        input_tensor = preprocess(image)

        if torch.mps.is_available():
            # print("Using MPS")
            device = torch.device("mps")
            # device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            print("Using cpu")

        input_batch = input_tensor.unsqueeze(0).to(device)
        model.to(device)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # for index in range(len(probabilities))
        predicted, predicted_indices = torch.topk(probabilities, 5)

        categories = weights.meta["categories"]
        print("Top 5 Predicted Categories:")
        print(f"Ground truth: {ground_truth}")
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        total += 1
        for i in range(top5_prob.size(0)):
            predicted_index = top5_indices[i].item()
            predicted_label = categories[predicted_index]
            if predicted_label == ground_truth:
                if i == 0:
                    accurate += 1
                total_prob += top5_prob[i].item()
            probability = top5_prob[i].item()
            print(f"  {i + 1}. **{predicted_label}** (Probability: {probability:.4f})")
        print("-" * 45)

    print(f"Accuracy is: {accurate / total}")

    print(f"average confidence is: {total_prob / total}")


classify_dataset("dataset/unprocessed", video=True)

# classify_dataset("dataset/xue", video=False)
#
# classify_dataset("dataset/xue2025-12-09T10:45:51", video=False)

# classify_dataset("dataset/average2025-12-09T11:44:31", video=False)
#
# classify_dataset("dataset/blur2025-12-09T11:44:58", video=False)
