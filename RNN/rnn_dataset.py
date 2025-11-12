import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import LATENT_DIM


class RNN_Dataset(Dataset):
    def __init__(self, data_path, seq_len=100, train=True, train_split_ratio=0.9):

        print(f"Loading preprocessed data from {data_path}...")
        with np.load(data_path, allow_pickle=True) as data:
            all_mus = data['mus']
            all_logvars = data['logvars']
            all_actions = data['actions']
            all_rewards = data['rewards']
            all_dones = data['dones']

        num_rollouts = len(all_mus)
        split_idx = int(num_rollouts * train_split_ratio)

        if train:
            self.mus = all_mus[:split_idx]
            self.logvars = all_logvars[:split_idx]
            self.actions = all_actions[:split_idx]
            self.rewards = all_rewards[:split_idx]
            self.dones = all_dones[:split_idx]
            print(f"Using {len(self.mus)} rollouts for TRAINING.")
        else:
            self.mus = all_mus[split_idx:]
            self.logvars = all_logvars[split_idx:]
            self.actions = all_actions[split_idx:]
            self.rewards = all_rewards[split_idx:]
            self.dones = all_dones[split_idx:]
            print(f"Using {len(self.mus)} rollouts for VALIDATION.")

        self.seq_len = seq_len
        self.indices = []

        for r_idx, (mus, actions) in enumerate(tqdm(zip(self.mus, self.actions))):
            num_frames = len(mus)
            for i in range(num_frames - seq_len - 1):
                self.indices.append((r_idx, i))

        print(f"Total number of sequences: {len(self.indices)}")

    def __reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rollout_idx, start_idx = self.indices[idx]

        mu_seq = torch.from_numpy(self.mus[rollout_idx][start_idx: start_idx + self.seq_len])
        logvar_seq = torch.from_numpy(self.logvars[rollout_idx][start_idx: start_idx + self.seq_len])

        z_t = self.__reparameterize(mu_seq, logvar_seq)

        a_t = torch.from_numpy(self.actions[rollout_idx][start_idx: start_idx + self.seq_len]).float()

        mu_next_seq = torch.from_numpy(self.mus[rollout_idx][start_idx + 1: start_idx + self.seq_len + 1])
        logvar_next_seq = torch.from_numpy(self.logvars[rollout_idx][start_idx + 1: start_idx + self.seq_len + 1])

        z_tplus1 = self.__reparameterize(mu_next_seq, logvar_next_seq)

        r_tplus1 = torch.from_numpy(self.rewards[rollout_idx][start_idx + 1: start_idx + self.seq_len + 1]).float()
        d_tplus1 = torch.from_numpy(self.dones[rollout_idx][start_idx + 1: start_idx + self.seq_len + 1]).float()

        return (z_t, a_t, z_tplus1, r_tplus1, d_tplus1)