import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with h5py.File(file_path, "r") as hf:
            image = np.array(hf["image"], dtype=np.float32)  # Load MRI image
            label = np.array(hf["mask"], dtype=np.int64)  # Load segmentation mask

        # Ensure correct format (C, H, W, D)
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        sample = {"image": torch.tensor(image), "label": torch.tensor(label)}

        if self.transform:
            sample = self.transform(sample)

        return sample
