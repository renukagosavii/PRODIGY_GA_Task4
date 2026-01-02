print("Script is running!")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# -----------------------------
# 1️⃣ Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_img = Image.open(os.path.join(self.input_dir, self.input_images[idx])).convert("RGB")
        target_img = Image.open(os.path.join(self.target_dir, self.target_images[idx])).convert("RGB")
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img

# -----------------------------
# 2️⃣ Transform
# ---
