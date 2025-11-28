import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RNN.rnn_model import MDN_RNN, mdn_loss, LATENT_DIM, ACTION_DIM, HIDDEN_DIM


# --- EFFICIENT DATASET CLASS ---
class MMapDataset(Dataset):
    def __init__(self, data_dir, seq_len=100, train=True):
        self.seq_len = seq_len

        # Load data in 'Read-Only' Memory Map mode
        self.mus = np.load(os.path.join(data_dir, 'mus.npy'), mmap_mode='r')
        self.logvars = np.load(os.path.join(data_dir, 'logvars.npy'), mmap_mode='r')
        self.actions = np.load(os.path.join(data_dir, 'actions.npy'), mmap_mode='r')
        self.rewards = np.load(os.path.join(data_dir, 'rewards.npy'), mmap_mode='r')
        self.dones = np.load(os.path.join(data_dir, 'dones.npy'), mmap_mode='r')

        boundaries = np.load(os.path.join(data_dir, 'boundaries.npy'))

        # Split rollouts for Train/Val
        num_rollouts = len(boundaries)
        split_idx = int(num_rollouts * 0.9)

        if train:
            active_boundaries = boundaries[:split_idx]
        else:
            active_boundaries = boundaries[split_idx:]

        print(f"Calculating valid sequences for {'TRAIN' if train else 'VAL'}...")
        self.valid_indices = []
        for start, length in active_boundaries:
            if length > seq_len + 1:
                end_idx = start + length - seq_len - 1
                self.valid_indices.extend(range(start, end_idx))

        print(f"Found {len(self.valid_indices)} valid sequences.")

    def __len__(self):
        return len(self.valid_indices)

    def __reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len

        # Copy small batch from disk to RAM
        mu_seq = torch.from_numpy(self.mus[start_idx: end_idx].copy())
        logvar_seq = torch.from_numpy(self.logvars[start_idx: end_idx].copy())
        z_t = self.__reparameterize(mu_seq, logvar_seq)

        a_t = torch.from_numpy(self.actions[start_idx: end_idx].copy())

        mu_next = torch.from_numpy(self.mus[start_idx + 1: end_idx + 1].copy())
        logvar_next = torch.from_numpy(self.logvars[start_idx + 1: end_idx + 1].copy())
        z_next = self.__reparameterize(mu_next, logvar_next)

        r_next = torch.from_numpy(self.rewards[start_idx + 1: end_idx + 1].copy())
        d_next = torch.from_numpy(self.dones[start_idx + 1: end_idx + 1].copy().astype(np.float32))

        return z_t, a_t, z_next, r_next, d_next


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/rnn_lpips_mmap_dataset', help='Path to MMAP directory')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ckpt_dir', type=str, default='rnn_checkpoints_lpips',
                        help='Folder to save per-epoch checkpoints')
    parser.add_argument('--log_dir', type=str, default='rnn_logs_lpips', help='Folder to save CSV logs')
    parser.add_argument('--save_model', type=str, default='rnn_lpips_final.pth')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_dataset = MMapDataset(args.data_path, seq_len=args.seq_len, train=True)
    val_dataset = MMapDataset(args.data_path, seq_len=args.seq_len, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    training_logs = []

    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        train_mdn = 0
        train_rew = 0
        train_done = 0
        train_acc = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for z, a, z_n, r, d in pbar:
            z, a, z_n = z.to(device), a.to(device), z_n.to(device)
            r, d = r.to(device), d.to(device)

            optimizer.zero_grad()

            (log_pi, mu, log_sigma, pred_r, pred_d), _ = model(z, a)

            l_mdn = mdn_loss(log_pi, mu, log_sigma, z_n)
            l_rew = F.mse_loss(pred_r.squeeze(-1), r)
            l_done = F.binary_cross_entropy_with_logits(pred_d.squeeze(-1), d)

            # Loss scaling
            loss = l_mdn + (20.0 * l_rew) + (20.0 * l_done)

            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            train_mdn += l_mdn.item()
            train_rew += l_rew.item()
            train_done += l_done.item()

            # Done Accuracy
            done_pred_binary = (torch.sigmoid(pred_d.squeeze(-1)) > 0.5).float()
            train_acc += (done_pred_binary == d).float().mean().item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_mdn = 0
        val_rew = 0
        val_done = 0
        val_acc = 0

        with torch.no_grad():
            for z, a, z_n, r, d in val_loader:
                z, a, z_n = z.to(device), a.to(device), z_n.to(device)
                r, d = r.to(device), d.to(device)

                (log_pi, mu, log_sigma, pred_r, pred_d), _ = model(z, a)

                l_mdn = mdn_loss(log_pi, mu, log_sigma, z_n)
                l_rew = F.mse_loss(pred_r.squeeze(-1), r)
                l_done = F.binary_cross_entropy_with_logits(pred_d.squeeze(-1), d)

                loss = l_mdn + (20.0 * l_rew) + (20.0 * l_done)

                val_loss += loss.item()
                val_mdn += l_mdn.item()
                val_rew += l_rew.item()
                val_done += l_done.item()

                done_pred_binary = (torch.sigmoid(pred_d.squeeze(-1)) > 0.5).float()
                val_acc += (done_pred_binary == d).float().mean().item()

        # --- AGGREGATE STATS ---
        n_train = len(train_loader)
        n_val = len(val_loader)

        t_loss = train_loss / n_train
        v_loss = val_loss / n_val
        v_acc = val_acc / n_val

        print(f"====> Epoch: {epoch}")
        print(f"      Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(
            f"      Val Breakdown -> MDN: {(val_mdn / n_val):.4f} | Reward MSE: {(val_rew / n_val):.5f} | Done Acc: {v_acc:.2%}")

        # Log Data
        training_logs.append({
            'epoch': epoch,
            'train_loss': t_loss,
            'val_loss': v_loss,
            'train_mdn': train_mdn / n_train,
            'val_mdn': val_mdn / n_val,
            'train_rew': train_rew / n_train,
            'val_rew': val_rew / n_val,
            'train_done_acc': train_acc / n_train,
            'val_done_acc': v_acc,
        })

        # Save CSV
        log_df = pd.DataFrame(training_logs)
        log_df.to_csv(os.path.join(args.log_dir, 'training_logs.csv'), index=False)

        # Save Checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f'rnn_epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"      Saved checkpoint: {ckpt_path}\n")

    # Save final model
    torch.save(model.state_dict(), args.save_model)
    print("Training Complete.")