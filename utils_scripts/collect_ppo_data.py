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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data using trained PPO agent")

    parser.add_argument('--num_rollouts', type=int, default=1000, help='Number of rollouts to collect')
    parser.add_argument('--data_dir', type=str, default='data/rollouts_v2_official', help='Directory to save .npz files')

    parser.add_argument('--vae_path', type=str, default='material_final/pesos/entrenamiento_1/vae_epoch_19.pth', help='Path to trained VAE model')
    parser.add_argument('--rnn_path', type=str, default='material_final/pesos/entrenamiento_1/rnn_epoch_1.pth', help='Path to trained RNN model')
    parser.add_argument('--ppo_path', type=str, default='material_final/pesos/entrenamiento_1/ppo_model_325000_steps.zip', help='Path to trained PPO model .zip')

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Collecting {args.num_rollouts} rollouts using PPO agent...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.eval()

    # CHANGE: Use standard MDN_RNN (matches your checkpoint with reward heads)
    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    print(f"Loading RNN from {args.rnn_path}")
    rnn.load_state_dict(torch.load(args.rnn_path, map_location=device))
    rnn.eval()

    print(f"Loading PPO model from {args.ppo_path}")
    model = PPO.load(args.ppo_path, device=device)

    env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])

    for i in tqdm(range(args.num_rollouts)):

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

            # Use deterministic=True for "Expert" behavior
            action, _ = model.predict(state_input, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Save data (Raw frame for VAE training)
            # Use transform but keep as numpy uint8 for storage efficiency
            frame_to_save = (transform(obs[:84, :, :]).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            observations.append(frame_to_save)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            next_z = process_frame(next_obs, transform, device, vae)

            action_tensor = torch.tensor(action, dtype=torch.float32).to(device).view(1, 1, ACTION_DIM)
            z_tensor = current_z.view(1, 1, LATENT_DIM)

            with torch.no_grad():
                # CHANGE: Unpack the extra return values (reward, done)
                # The standard MDN_RNN returns: (pi, mu, sigma, r, d), hidden_state
                (_, _, _, _, _), hidden_state = rnn(z_tensor, action_tensor, hidden_state)

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
    print(f"\nData collection complete.")