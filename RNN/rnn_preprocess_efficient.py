import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
import sys
import gc  # Garbage Collector

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM

# Settings
BATCH_SIZE = 64  # Process frames in chunks to save VRAM
RESIZE_SIZE = 64


def preprocess_efficient(data_dir, output_path, vae_path, max_rollouts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target Latent Dim: {LATENT_DIM}")

    # 1. Load VAE
    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()

    # 2. Setup Transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])

    # 3. Get File List
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if max_rollouts > 0:
        all_files = all_files[:max_rollouts]

    print(f"Found {len(all_files)} rollouts to process.")

    # 4. Storage Lists (We only store the small vectors here)
    all_mus = []
    all_logvars = []
    all_actions = []
    all_rewards = []
    all_dones = []

    # 5. Loop with Memory Management
    print("Starting processing... (One rollout at a time)")

    for f in tqdm(all_files):
        # A. Load ONE rollout into memory
        try:
            with np.load(f) as data:
                # Load strictly what we need
                observations = data['observations']  # Heavy (Images)
                actions = data['actions']
                rewards = data['rewards']
                dones = data['dones']
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

        mu_sequence = []
        logvar_sequence = []

        # B. Process frames in batches (VRAM safety)
        # We use torch.no_grad() to save RAM/VRAM
        with torch.no_grad():
            total_frames = len(observations)
            for i in range(0, total_frames, BATCH_SIZE):
                obs_batch_np = observations[i: i + BATCH_SIZE]

                # Convert to Tensor
                obs_batch_tensors = []
                for frame in obs_batch_np:
                    obs_batch_tensors.append(transform(frame))

                obs_batch_in = torch.stack(obs_batch_tensors).to(device)

                # Encode
                mu, logvar = vae.encode(obs_batch_in)

                # Store only the vectors (CPU)
                mu_sequence.append(mu.cpu().numpy())
                logvar_sequence.append(logvar.cpu().numpy())

        # C. Append results to main lists
        all_mus.append(np.concatenate(mu_sequence, axis=0))
        all_logvars.append(np.concatenate(logvar_sequence, axis=0))
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)

        # D. CRITICAL: Free Memory immediately
        del observations
        del actions
        del rewards
        del dones
        del mu_sequence
        del logvar_sequence

        # Force Python to release memory
        gc.collect()

    # 6. Save Final Dataset
    print(f"Processing complete. Saving compressed dataset to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        mus=np.array(all_mus, dtype=object),
        logvars=np.array(all_logvars, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
        dones=np.array(all_dones, dtype=object)
    )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/rollouts_v2', help='Directory of raw .npz rollouts')
    parser.add_argument('--vae_model', type=str, default='model_checkpoints/vae_2_checkpoints/vae_epoch_20.pth', help='Path to VAE checkpoint')
    parser.add_argument('--output_path', type=str, default='data/rnn_2_dataset/rnn_dataset.npz')
    parser.add_argument('--max_rollouts', type=int, default=1000)

    args = parser.parse_args()
    preprocess_efficient(args.data_dir, args.output_path, args.vae_model, args.max_rollouts)