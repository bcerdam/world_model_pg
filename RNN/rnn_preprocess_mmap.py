import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM

BATCH_SIZE = 128
RESIZE_SIZE = 64


def preprocess_mmap(data_dir, output_dir, vae_path, max_rollouts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load VAE
    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])

    all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if max_rollouts > 0:
        all_files = all_files[:max_rollouts]

    print(f"Found {len(all_files)} rollouts. Calculating total size...")

    # 2. First Pass: Calculate total frames
    total_frames = 0
    for f in tqdm(all_files, desc="Scanning files"):
        try:
            with np.load(f) as data:
                n_frames = len(data['rewards'])
                total_frames += n_frames
        except:
            continue

    print(f"Total dataset size: {total_frames} frames.")

    os.makedirs(output_dir, exist_ok=True)

    # 3. Create Valid .npy Files on Disk
    # --- THE FIX: Use np.lib.format.open_memmap ---
    # This creates a file with a valid .npy header, so np.load() works later.
    print("Allocating disk space...")
    fp_mu = np.lib.format.open_memmap(os.path.join(output_dir, 'mus.npy'), mode='w+', dtype='float32',
                                      shape=(total_frames, LATENT_DIM))
    fp_logvar = np.lib.format.open_memmap(os.path.join(output_dir, 'logvars.npy'), mode='w+', dtype='float32',
                                          shape=(total_frames, LATENT_DIM))
    fp_action = np.lib.format.open_memmap(os.path.join(output_dir, 'actions.npy'), mode='w+', dtype='float32',
                                          shape=(total_frames, 3))
    fp_reward = np.lib.format.open_memmap(os.path.join(output_dir, 'rewards.npy'), mode='w+', dtype='float32',
                                          shape=(total_frames,))
    fp_done = np.lib.format.open_memmap(os.path.join(output_dir, 'dones.npy'), mode='w+', dtype='bool',
                                        shape=(total_frames,))

    # 4. Second Pass: Fill the arrays
    current_idx = 0
    rollout_boundaries = []

    print("Processing and writing to disk...")
    for f in tqdm(all_files):
        try:
            with np.load(f) as data:
                obs = data['observations']
                act = data['actions']
                rew = data['rewards']
                don = data['dones']
        except:
            continue

        n_frames = len(obs)

        # Save simple data
        fp_action[current_idx: current_idx + n_frames] = act
        fp_reward[current_idx: current_idx + n_frames] = rew
        fp_done[current_idx: current_idx + n_frames] = don

        # Process VAE in batches
        with torch.no_grad():
            for i in range(0, n_frames, BATCH_SIZE):
                batch_obs = obs[i: i + BATCH_SIZE]

                # Preprocess batch
                batch_tensors = torch.stack([transform(o) for o in batch_obs]).to(device)

                # Encode
                mu, logvar = vae.encode(batch_tensors)

                # Write to disk
                end_i = min(i + BATCH_SIZE, n_frames)
                fp_mu[current_idx + i: current_idx + end_i] = mu.cpu().numpy()
                fp_logvar[current_idx + i: current_idx + end_i] = logvar.cpu().numpy()

        rollout_boundaries.append((current_idx, n_frames))
        current_idx += n_frames

        del obs, act, rew, don
        gc.collect()

    # Flush changes to disk
    fp_mu.flush()
    fp_logvar.flush()
    fp_action.flush()
    fp_reward.flush()
    fp_done.flush()

    np.save(os.path.join(output_dir, 'boundaries.npy'), np.array(rollout_boundaries))

    print(f"Done! Dataset saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/rollouts_v2_official')
    parser.add_argument('--output_dir', type=str, default='data/rnn_32_mmap_dataset')
    parser.add_argument('--vae_model', type=str, default='material_final/pesos/entrenamiento_1_2/32/vae_epoch_40.pth')
    parser.add_argument('--max_rollouts', type=int, default=1000)

    args = parser.parse_args()
    preprocess_mmap(args.data_dir, args.output_dir, args.vae_model, args.max_rollouts)