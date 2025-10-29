import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#Define Paths
DATA_DIR = Path("data/plant_dataset/New Plant Diseases Dataset(Augmented)")
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = Path("data/test/test")

print(f"Train classes: {len(list(TRAIN_DIR.glob('*')))}")
print(f"Valid classes: {len(list(VALID_DIR.glob('*')))}")
print(f"Test images: {len(list(TEST_DIR.glob('*.JPG')))}")

#Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Load Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Valid batches: {len(valid_loader)}")
print(f"Classes: {train_dataset.classes[:5]}...")

#Visualize Sample
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

# Get one batch
dataiter = iter(train_loader)
images, labels = next(dataiter)

def imshow_save(img, title, path="sample_batch.png"):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title, fontsize=10)
    plt.axis('off')

# Create figure
fig = plt.figure(figsize=(12, 6))
# Show 6 images
for idx in range(6):  
    ax = fig.add_subplot(2, 3, idx+1)
    imshow_save(images[idx], train_dataset.classes[labels[idx]])
plt.tight_layout()
plt.savefig("sample_batch.png", dpi=150, bbox_inches='tight')
print("Sample batch saved as 'sample_batch.png'")
plt.close()