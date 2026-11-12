import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

# Fix random seed to improve experiment reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name(0))

# hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

NUM_CLASSES = 10
TRAIN_SIZE = 45000
VALID_SIZE = 5000

DATA_DIR = "data"
CHECKPOINT_PATH = "best_vgg19_cifar10.pth"

NUM_WORKERS = 2 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# # CIFAR10 URL
# datasets.CIFAR10.url = (
#     "https://huggingface.co/datasets/MIT-OL-AI-D/"
#     "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
# )

# cifar_trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())

# imgs = [item[0] for item in cifar_trainset] # item[0] and item[1] are image and its label
# imgs = torch.stack(imgs, dim=0).numpy()

# # calculate mean over each channel (r,g,b)
# mean_r = imgs[:,0,:,:].mean()
# mean_g = imgs[:,1,:,:].mean()
# mean_b = imgs[:,2,:,:].mean()
# print(mean_r,mean_g,mean_b)

# # calculate std over each channel (r,g,b)
# std_r = imgs[:,0,:,:].std()
# std_g = imgs[:,1,:,:].std()
# std_b = imgs[:,2,:,:].std()
# print(std_r,std_g,std_b)

### Data Preprocessing ###

# CIFAR10 URL
datasets.CIFAR10.url = (
    "https://huggingface.co/datasets/MIT-OL-AI-D/"
    "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
)

# CIFAR-10 MEAN and STD over each channel (r,g,b)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# train data: Data Augmentation + Normalize
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# valid data
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

train_data_aug = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    transform=train_transform,
    download=True
)

train_data_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    transform=eval_transform,
    download=True
)

# test data
test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    transform=eval_transform,
    download=True
)

# Split Train / Valid Data, ensuring no overlapping images
generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(train_data_aug),
    generator=generator
).tolist()

train_indices = indices[:TRAIN_SIZE]
valid_indices = indices[TRAIN_SIZE:TRAIN_SIZE + VALID_SIZE]

train_dataset = Subset(train_data_aug, train_indices)
valid_dataset = Subset(train_data_eval, valid_indices)

assert len(train_dataset) == TRAIN_SIZE
assert len(valid_dataset) == VALID_SIZE
assert len(test_dataset) == 10000
assert set(train_indices).isdisjoint(set(valid_indices))

print(f"Train images: {len(train_dataset)}")
print(f"Valid images: {len(valid_dataset)}")
print(f"Test  images:  {len(test_dataset)}")

# dataLoader
loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    generator=loader_generator
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

class VGG19(torch.nn.Module):
    def __init__(self,num_classes):
        super(VGG19, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block5  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.classifier(x.view(-1,512*1*1))
        probas = F.softmax(logits,dim = 1)
        return logits,probas

model = VGG19(num_classes = NUM_CLASSES)
model = model.to(DEVICE)

print(model)
# print(
#     "Trainable parameters:",
#     sum(p.numel() for p in model.parameters() if p.requires_grad)
# )

# Optimizer: Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Learning Rate * 0.5
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

# Compute accuracy and cross entropy loss
def compute_accuracy_and_loss(model, data_loader, device):
    model.eval()

    correct_pred = 0
    num_examples = 0
    total_loss = 0.0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits, _ = model(features)

            loss = F.cross_entropy(
                logits,
                targets,
                reduction="sum"
            )

            total_loss += loss.item()

            predicted_labels = logits.argmax(dim=1)

            correct_pred += (
                predicted_labels == targets
            ).sum().item()

            num_examples += targets.size(0)

    avg_loss = total_loss / num_examples
    accuracy = correct_pred / num_examples * 100

    return accuracy, avg_loss

### training ###

best_valid_loss = float("inf")

history = {
    "train_loss": [],
    "train_acc": [],
    "valid_loss": [],
    "valid_acc": [],
    "lr": []
}

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    current_lr = optimizer.param_groups[0]["lr"]

    model.train()

    running_loss = 0.0
    correct_pred = 0
    num_examples = 0

    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits, _ = model(features)

        cost = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        cost.backward()
        optimizer.step()

        batch_size = targets.size(0)

        running_loss += cost.item() * batch_size
        predicted_labels = logits.argmax(dim=1)

        correct_pred += (
            predicted_labels == targets
        ).sum().item()

        num_examples += batch_size

        if (batch_idx + 1) % 300 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {cost.item():.4f}"
            )

    train_loss = running_loss / num_examples
    train_acc = correct_pred / num_examples * 100

    valid_acc, valid_loss = compute_accuracy_and_loss(
        model,
        valid_loader,
        DEVICE
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)
    history["lr"].append(current_lr)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        best_marker = "  <-- Best model saved"
    else:
        best_marker = ""

    elapsed = time.time() - epoch_start

    print(
        f"\nEpoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"Valid Acc: {valid_acc:.2f}% | "
        f"Time: {elapsed:.1f}s"
        f"{best_marker}\n"
    )

    scheduler.step()

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE)
)

test_acc, test_loss = compute_accuracy_and_loss(
    model,
    test_loader,
    DEVICE
)

print("=" * 60)
print("Final Test Results (Best Validation Model)")
print(f"Best Validation Loss: {best_valid_loss:.4f}")
print(f"Test Accuracy:        {test_acc:.2f}%")
print(f"Test Loss:            {test_loss:.4f}")
print(f"Saved model:          {CHECKPOINT_PATH}")
print("=" * 60)

epochs = range(1, NUM_EPOCHS + 1)

# Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["valid_loss"], label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# Accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_acc"], label="Train Accuracy")
plt.plot(epochs, history["valid_acc"], label="Valid Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.show()