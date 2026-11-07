from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import Dataset

class ShipScopeDataset(Dataset):
    def __init__(self, directory, preprocessing=None):
        self.preprocessing = preprocessing
        self.paths = sorted(Path(directory).glob('*.png'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(path.name.split("__", 1)[0])

        img = cv.imread(str(path))

        if img is None:
            raise RuntimeError(f"Couldn't load image from {path}")

        if img.shape != (80, 80, 3):
            raise ValueError(f"Unexpected image shape {img.shape} in {path}")

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.preprocessing is not None:
            for method in self.preprocessing:
                img = method(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label)

        return img, label