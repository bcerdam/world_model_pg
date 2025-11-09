import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class RNN_Dataset(Dataset):
    def __init__(self, data_path, seq_len=100, train=True, train_split_ratio=0.9):

        print(f"Loading preprocessed data from {data_path}...")
        with np.load(data_path, allow_pickle=True) as data:
            all_latents = data['latents']
            all_actions = data['actions']

        num_rollouts = len(all_latents)
        split_idx = int(num_rollouts * train_split_ratio)

        if train:
            self.latents = all_latents[:split_idx]
            self.actions = all_actions[:split_idx]
            print(f"Using {len(self.latents)} rollouts for TRAINING.")
        else:
            self.latents = all_latents[split_idx:]
            self.actions = all_actions[split_idx:]
            print(f"Using {len(self.latents)} rollouts for VALIDATION.")

        self.seq_len = seq_len
        self.indices = []

        for r_idx, (latents, actions) in enumerate(tqdm(zip(self.latents, self.actions))):
            num_frames = len(latents)
            for i in range(num_frames - seq_len - 1):
                self.indices.append((r_idx, i))

        print(f"Total number of sequences: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rollout_idx, start_idx = self.indices[idx]

        z_t = self.latents[rollout_idx][start_idx: start_idx + self.seq_len]
        a_t = self.actions[rollout_idx][start_idx: start_idx + self.seq_len]

        z_tplus1 = self.latents[rollout_idx][start_idx + 1: start_idx + self.seq_len + 1]

        return (
            torch.from_numpy(z_t).float(),
            torch.from_numpy(a_t).float(),
            torch.from_numpy(z_tplus1).float()
        )