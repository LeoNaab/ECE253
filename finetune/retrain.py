import os
import time
from multiprocessing import Value
from tempfile import TemporaryDirectory

import torch
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)

preprocess = weights.transforms()

DATA_PATH = "./data"
BATCH_SIZE = 8
LEARNING_RATE = 0.0001  #
EPOCHS = 10

full_dataset = datasets.ImageFolder(DATA_PATH, transform=preprocess)
print(f"Your Local Classes: {full_dataset.classes}")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = "data/"
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=0
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

if torch.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")
    # device = torch.device("cpu")
else:
    device = torch.device("cpu")
    print("Using cpu")


model.to(device)


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    dataset_sizes,
    device,
    train_map,
    val_map,
    num_epochs=2,
):
    since = time.time()

    tempdir = "temp"
    os.makedirs(tempdir, exist_ok=True)
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                map = train_map
            else:
                model.eval()  # Set model to evaluate mode
                map = val_map

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                mapped_labels = []
                for label in labels:
                    mapped_labels.append(map[label.item()])
                labels = torch.tensor(mapped_labels).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == "train":
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def build_map(dataset, set_name):
    """Creates a map for a specific dataset"""
    mapping = {}
    print(f"{set_name} Set ---")

    for local_idx, class_name in enumerate(dataset.classes):
        clean_name = class_name.replace("_", " ").lower()

        try:
            global_idx = imagenet_classes.index(clean_name)
            mapping[local_idx] = global_idx
            print(f"local id {local_idx} ('{class_name}') -> imagenet id {global_idx}")
        except ValueError:
            print(f"'{class_name}' not found in imagenet list")

    return mapping


imagenet_classes = weights.meta["categories"]
local_to_global_map = {}
print("\n--- Building Index Map ---")

train_map = build_map(image_datasets["train"], "Train")
val_map = build_map(image_datasets["val"], "Val")

optimizer = torch.optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# Freeze Backbone (Optional: set to True if you want to update backbone too)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the 1000-class head so we can fine-tune the specific indices
for param in model.fc.parameters():
    param.requires_grad = True

train_model(
    model,
    criterion,
    optimizer,
    dataloaders=dataloaders,
    dataset_sizes=dataset_sizes,
    device=device,
    train_map=train_map,
    val_map=val_map,
    num_epochs=2,
)
