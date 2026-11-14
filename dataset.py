import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Task 1: Explicit Data Contract matching official class_dict.csv
CLASS_NAMES = [
    "Informal Settlements",  # ID 0: RGB (250, 235, 185)
    "Built-Up",              # ID 1: RGB (200, 200, 200)
    "Impervious Surfaces",   # ID 2: RGB (100, 100, 150)
    "Vegetation",            # ID 3: RGB (80, 140, 50)
    "Barren",                # ID 4: RGB (200, 160, 40)
    "Water"                  # ID 5: RGB (40, 120, 240)
]

CLASS_COLORS = np.array([
    [250, 235, 185],  # 0: Informal Settlements
    [200, 200, 200],  # 1: Built-Up
    [100, 100, 150],  # 2: Impervious Surfaces
    [80, 140, 50],    # 3: Vegetation
    [200, 160, 40],   # 4: Barren
    [40, 120, 240]    # 5: Water
], dtype=np.uint8)

# Official 'Unlabelled' (0, 0, 0) mapped to IGNORE_INDEX = 255
IGNORE_INDEX = 255
IMG_SIZE = (256, 256)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def rgb_to_mask_with_ignore(rgb_np):
    """
    Maps RGB mask (H, W, 3) to integer class IDs [0..5].
    Pixels with RGB (0, 0, 0) ('Unlabelled') or transition artifacts are mapped to 255.
    """
    h, w, _ = rgb_np.shape
    mask = np.full((h, w), IGNORE_INDEX, dtype=np.int64)

    # Unlabelled class in class_dict.csv is (0, 0, 0)
    is_unlabelled = np.all(rgb_np < 15, axis=-1)

    # Nearest L1 color distance to the 6 valid classes
    diffs = np.abs(rgb_np[:, :, None, :].astype(np.int32) - CLASS_COLORS[None, None, :, :].astype(np.int32)).sum(axis=-1)
    nearest_cls = np.argmin(diffs, axis=-1)
    min_dist = np.min(diffs, axis=-1)

    valid = (~is_unlabelled) & (min_dist < 45)
    mask[valid] = nearest_cls[valid]
    return mask

class MumbaiSatelliteDataset(Dataset):
    def __init__(self, pairs, split="train", img_size=IMG_SIZE):
        self.pairs = pairs
        self.split = split
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        # Transform Contract: Bilinear for RGB image, Nearest-Neighbor for mask
        image = TF.resize(image, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask_rgb = TF.resize(mask_rgb, self.img_size, interpolation=InterpolationMode.NEAREST)

        # Synchronized geometric augmentations for training split only
        if self.split == "train":
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask_rgb = TF.hflip(mask_rgb)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask_rgb = TF.vflip(mask_rgb)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask_rgb = TF.rotate(mask_rgb, angle, interpolation=InterpolationMode.NEAREST)

        img_tensor = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD)
        mask_ids = rgb_to_mask_with_ignore(np.array(mask_rgb, dtype=np.uint8))
        mask_tensor = torch.from_numpy(mask_ids).long()

        return img_tensor, mask_tensor, os.path.basename(img_path)

def get_mumbai_dataloaders(data_root="mumbai_seg_exp/data/Dataset/Main_Dataset", batch_size=8, seed=42):
    img_files = sorted(glob.glob(os.path.join(data_root, "images", "*.*")))
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p 
                for p in glob.glob(os.path.join(data_root, "masks", "*.*"))}

    all_pairs = [(p, mask_map[os.path.splitext(os.path.basename(p))[0]]) 
                 for p in img_files if os.path.splitext(os.path.basename(p))[0] in mask_map]

    # Deterministic 70% Train (77) / 15% Val (16) / 15% Test (17) Split
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    train_ds = MumbaiSatelliteDataset(all_pairs[:n_train], split="train")
    val_ds   = MumbaiSatelliteDataset(all_pairs[n_train:n_train+n_val], split="val")
    test_ds  = MumbaiSatelliteDataset(all_pairs[n_train+n_val:], split="test")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
