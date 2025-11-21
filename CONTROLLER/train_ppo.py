import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import os
import sys

# --- Add project root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64
WARMUP_STEPS = 50


class WorldModelEnv(gym.Env):
    def __init__(self, vae_path, rnn_path, device):
        super(WorldModelEnv, self).__init__()

        self.device = device
        self.vae = VAE(z_dim=LATENT_DIM).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.vae.eval()

        self.rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(self.device)
        self.rnn.load_state_dict(torch.load(rnn_path, map_location=self.device), strict=False)
        self.rnn.eval()

        self.real_env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)

        self.action_space = self.real_env.action_space

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(LATENT_DIM + HIDDEN_DIM,), dtype=np.float32
        )

        # WHERE THE CHANGE IS: Removed T.Crop.
        # We only use ToPILImage (to handle numpy input), Resize, and ToTensor.
        # Cropping happens manually in _process_frame.
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])

        self.current_z = None
        self.hidden_state = None

    def _process_frame(self, frame):
        # WHERE THE CHANGE IS: Manual numpy slicing to crop the dashboard.
        # Keep top 84 pixels, full width, all channels.
        frame = frame[:84, :, :]

        frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, logvar = self.vae.encode(frame_tensor)
            z = self.vae.reparameterize(mu, logvar)
        return z

    def reset(self, seed=None, options=None):
        obs, info = self.real_env.reset(seed=seed, options=options)

        warmup_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(WARMUP_STEPS):
            obs, _, _, _, _ = self.real_env.step(warmup_action)

        self.current_z = self._process_frame(obs)
        self.hidden_state = None

        h_zeros = torch.zeros(1, HIDDEN_DIM).to(self.device)

        state = torch.cat([self.current_z.squeeze(0), h_zeros.squeeze(0)], dim=0).cpu().numpy()
        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.real_env.step(action)

        next_z = self._process_frame(obs)

        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device).view(1, 1, ACTION_DIM)
        z_tensor = self.current_z.view(1, 1, LATENT_DIM)

        with torch.no_grad():
            _, self.hidden_state = self.rnn(z_tensor, action_tensor, self.hidden_state)

        h_t = self.hidden_state[0].squeeze(0).squeeze(0)

        next_state = torch.cat([next_z.squeeze(0), h_t], dim=0).cpu().numpy()

        self.current_z = next_z

        # if reward < 0:
        #     reward += 0.1

        return next_state, reward, terminated, truncated, info

    def close(self):
        self.real_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- YOUR CUSTOM PATHS ---
    parser.add_argument('--vae_path', type=str,
                        default='model_checkpoints/vae_2_checkpoints/vae_epoch_19.pth')
    parser.add_argument('--rnn_path', type=str,
                        default='rnn_2_checkpoints/rnn_epoch_1.pth')
    parser.add_argument('--save_path', type=str, default='model_checkpoints/controller_2_checkpoints/ppo_car_racing')
    parser.add_argument('--total_timesteps', type=int, default=3000000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PPO on device: {device}")

    env = WorldModelEnv(args.vae_path, args.rnn_path, device)

    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./ppo_checkpoints/',
        name_prefix='ppo_model'
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.01,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device=device
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")