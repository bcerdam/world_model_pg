import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from stable_baselines3 import PPO
from tqdm import tqdm
import argparse
import os
import sys
import torch.multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64
WARMUP_STEPS = 50


def process_frame(frame, transform, device, vae):
    # 1. Crop dashboard
    frame = frame[:84, :, :]

    # 2. Transform to Tensor
    frame_tensor = transform(frame).unsqueeze(0).to(device)

    # 3. Pass through VAE
    with torch.no_grad():
        mu, logvar = vae.encode(frame_tensor)
        z = vae.reparameterize(mu, logvar)
    return z


def collect_worker(gpu_id, rollout_indices, args):
    """
    Worker function to run rollouts on a specific GPU.
    """
    # Assign this process to a specific GPU
    device = torch.device(f"cuda:{gpu_id}")

    # --- Load Models (Must be loaded inside the process) ---
    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.eval()

    # Use standard MDN_RNN (assuming your checkpoint has the heads)
    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)

    # If your checkpoint is missing heads, use strict=False.
    # If it has them, strict=True (default) is fine.
    # We use strict=False to be safe for both cases.
    rnn.load_state_dict(torch.load(args.rnn_path, map_location=device), strict=False)
    rnn.eval()

    model = PPO.load(args.ppo_path, device=device)

    # --- Setup Environment ---
    env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])

    # --- Collection Loop ---
    # We use position=gpu_id to offset progress bars so they don't overlap visually
    for i in tqdm(rollout_indices, desc=f"GPU {gpu_id}", position=gpu_id):

        obs, info = env.reset()

        warmup_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(WARMUP_STEPS):
            obs, _, _, _, _ = env.step(warmup_action)

        current_z = process_frame(obs, transform, device, vae)

        hidden_state = None
        h_t = torch.zeros(1, HIDDEN_DIM).to(device)

        observations = []
        actions = []
        rewards = []
        dones = []

        done = False

        while not done:
            state_input = torch.cat([current_z.squeeze(0), h_t.squeeze(0)], dim=0).cpu().numpy()

            # Expert behavior: deterministic=True
            action, _ = model.predict(state_input, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Save data
            frame_to_save = (transform(obs[:84, :, :]).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            observations.append(frame_to_save)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            next_z = process_frame(next_obs, transform, device, vae)

            action_tensor = torch.tensor(action, dtype=torch.float32).to(device).view(1, 1, ACTION_DIM)
            z_tensor = current_z.view(1, 1, LATENT_DIM)

            with torch.no_grad():
                # Unpack 5 values, ignore the last 2 (reward/done)
                (log_pi, mu, log_sigma, _, _), hidden_state = rnn(z_tensor, action_tensor, hidden_state)

            h_t = hidden_state[0].squeeze(0).squeeze(0)
            current_z = next_z
            obs = next_obs

        filename = os.path.join(args.data_dir, f'rollout_{i}.npz')

        np.savez_compressed(
            filename,
            observations=np.array(observations, dtype=np.uint8),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=bool)
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data using trained PPO agent (Multi-GPU)")

    parser.add_argument('--num_rollouts', type=int, default=10000, help='Total number of rollouts to collect')
    parser.add_argument('--data_dir', type=str, default='data/rollouts_v2', help='Directory to save .npz files')
    parser.add_argument('--vae_path', type=str, default='model_checkpoints/vae_checkpoints/vae_epoch_10.pth',
                        help='Path to trained VAE model')
    parser.add_argument('--rnn_path', type=str, default='rnn_checkpoints/rnn_epoch_4.pth',
                        help='Path to trained RNN model')
    parser.add_argument('--ppo_path', type=str, default='ppo_dream_checkpoints/ppo_dream_1450000_steps.zip',
                        help='Path to trained PPO model .zip')

    args = parser.parse_args()

    # Ensure 'spawn' method for CUDA multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(args.data_dir, exist_ok=True)

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found! Falling back to CPU (single worker).")
        num_gpus = 1
        device_ids = [-1]  # Indicator for CPU
    else:
        print(f"Found {num_gpus} GPUs. Distributing work...")
        device_ids = range(num_gpus)

    # Split rollouts evenly across GPUs
    all_indices = list(range(args.num_rollouts))

    chunk_size = int(np.ceil(args.num_rollouts / num_gpus))
    processes = []

    for rank, gpu_id in enumerate(device_ids):
        start_idx = rank * chunk_size
        end_idx = min((rank + 1) * chunk_size, args.num_rollouts)

        worker_indices = all_indices[start_idx:end_idx]

        if not worker_indices:
            continue

        p = mp.Process(
            target=collect_worker,
            args=(gpu_id if gpu_id >= 0 else 'cpu', worker_indices, args)
        )
        p.start()
        processes.append(p)

    # Wait for all to finish
    for p in processes:
        p.join()

    print(f"\nData collection complete on all GPUs.")