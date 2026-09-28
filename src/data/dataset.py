"""Dataset loader for (image, mask, height) triples."""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMG_SIZE = 256
MAX_HEIGHT_M = 60.0


class SatelliteBuildingDataset(Dataset):
    """Reads (image, mask, height) triples from disk."""

    def __init__(
        self,
        root,
        ids,
        img_size=IMG_SIZE,
        augment=False,
        max_height_m=MAX_HEIGHT_M,
    ):
        self.root = root
        self.ids = ids
        self.img_size = img_size
        self.augment = augment
        self.max_height_m = max_height_m
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.ids)

    def _load_triple(self, sample_id):
        img = np.array(
            Image.open(os.path.join(self.root, "images", f"{sample_id}.png")).convert("RGB")
        )
        mask = np.array(
            Image.open(os.path.join(self.root, "masks", f"{sample_id}.png")).convert("L")
        )
        mask = (mask > 127).astype(np.float32)
        height = np.load(os.path.join(self.root, "heights", f"{sample_id}.npy")).astype(np.float32)
        return img, mask, height

    def _augment(self, img, mask, height):
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
            height = np.ascontiguousarray(height[:, ::-1])
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])
            height = np.ascontiguousarray(height[::-1, :])
        k = random.choice([0, 1, 2, 3])
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
            height = np.ascontiguousarray(np.rot90(height, k))
        return img, mask, height

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img, mask, height = self._load_triple(sample_id)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            height = cv2.resize(
                height, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
            )

        if self.augment:
            img, mask, height = self._augment(img, mask, height)

        img_t = TF.to_tensor(img.copy())
        img_t = TF.normalize(img_t, self.imagenet_mean, self.imagenet_std)
        mask_t = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        height_norm = np.clip(height / self.max_height_m, 0.0, 1.0)
        height_t = torch.from_numpy(height_norm.copy()).unsqueeze(0).float()

        return {"image": img_t, "mask": mask_t, "height": height_t, "id": sample_id}
