import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from stable_baselines3 import PPO
import imageio.v3 as iio
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM

ENV_NAME = 'CarRacing-v3'
RESIZE_SIZE = 64
WARMUP_STEPS = 50


# We must redefine the Env wrapper to load the model (or import it if you structured it as a module)
class WorldModelEnv(gym.Env):
    def __init__(self, vae_path, rnn_path, device):
        super(WorldModelEnv, self).__init__()
        self.device = device
        self.vae = VAE(z_dim=LATENT_DIM).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.vae.eval()
        self.rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(self.device)
        self.rnn.load_state_dict(torch.load(rnn_path, map_location=self.device))
        self.rnn.eval()
        self.real_env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)
        self.action_space = self.real_env.action_space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(LATENT_DIM + HIDDEN_DIM,),
                                                dtype=np.float32)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])
        self.current_z = None
        self.hidden_state = None

    def _process_frame(self, frame):
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
        return next_state, reward, terminated, truncated, info, obs

    def close(self):
        self.real_env.close()


def create_agent_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Environment
    env = WorldModelEnv(args.vae_path, args.rnn_path, device)

    # Load PPO Model
    print(f"Loading PPO model from: {args.ppo_path}")
    model = PPO.load(args.ppo_path, device=device)

    obs, _ = env.reset()
    frames = []
    total_reward = 0
    done = False

    print("Running agent...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Deterministic for evaluation

        # Note: modified step to return 'raw_obs' for video
        obs, reward, terminated, truncated, info, raw_frame = env.step(action)

        total_reward += reward
        frames.append(raw_frame)

        done = terminated or truncated

    env.close()

    print(f"Episode finished. Total Reward: {total_reward:.2f}")
    print(f"Saving video to {args.output_path}...")

    iio.imwrite(
        args.output_path,
        frames,
        fps=30,
        pixelformat='yuv420p'
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_path', type=str,
                        default='cluster_results/model_checkpoints/vae_checkpoints/vae_epoch_10.pth')
    parser.add_argument('--rnn_path', type=str,
                        default='cluster_results/model_checkpoints/rnn_checkpoints/rnn_epoch_4.pth')
    # Point this to your latest checkpoint or final model
    parser.add_argument('--ppo_path', type=str, required=True, help='Path to .zip PPO model file')
    parser.add_argument('--output_path', type=str, default='video_output/agent_run_non_dream.mp4')

    args = parser.parse_args()
    create_agent_video(args)