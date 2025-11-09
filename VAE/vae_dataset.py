import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm


class VAE_Dataset(Dataset):
    def __init__(self, data_dir, train=True, train_split_ratio=0.9, max_rollouts=2000):
        # WHERE THE CHANGE IS: This logic is new.
        # It now splits the file list into training and validation sets.
        all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))[:max_rollouts]
        split_idx = int(len(all_files) * train_split_ratio)

        if train:
            self.files = all_files[:split_idx]
            print(f"Loading {len(self.files)} rollouts for TRAINING...")
        else:
            self.files = all_files[split_idx:]
            print(f"Loading {len(self.files)} rollouts for VALIDATION...")

        self.observations = []
        for f in tqdm(self.files):
            with np.load(f) as data:
                self.observations.append(data['observations'])

        self.observations = np.concatenate(self.observations, axis=0)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])

        print(f"Total frames loaded: {len(self.observations)}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        frame = self.observations[idx]
        return self.transform(frame)