import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
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

# --- CONFIGURATION MAP ---
# Maps your experiment names to specific weight files
WEIGHT_PATHS = {
    "normal_latent": {
        "vae": "pesos_prueba_oficial/vae_rnn/vae_epoch_40.pth",
        "rnn": "pesos_prueba_oficial/vae_rnn/rnn_epoch_1.pth",
        "mode": "latent"  # Use z + h
    },
    "decoded_latent": {
        "vae": "pesos_prueba_oficial/vae_rnn/vae_epoch_40.pth",
        "rnn": "pesos_prueba_oficial/vae_rnn/rnn_epoch_1.pth",
        "mode": "visual"  # Use decoded(z) + h
    },
    "lpips_latent": {
        "vae": "pesos_prueba_oficial/vae_lpips_rnn/vae_epoch_50.pth",
        "rnn": "pesos_prueba_oficial/vae_lpips_rnn/rnn_epoch_1.pth",
        "mode": "latent"  # Use z + h (but from LPIPS weights)
    }
}


class FlexibleWorldModelEnv(gym.Env):
    def __init__(self, vae_path, rnn_path, mode, device):
        super(FlexibleWorldModelEnv, self).__init__()
        self.device = device
        self.mode = mode  # 'latent' or 'visual'

        # Load Models
        self.vae = VAE(z_dim=LATENT_DIM).to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        self.vae.eval()

        self.rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(self.device)
        # strict=False allows loading even if one has dropout and the other doesn't
        self.rnn.load_state_dict(torch.load(rnn_path, map_location=self.device), strict=False)
        self.rnn.eval()

        self.real_env = gym.make(ENV_NAME, render_mode='rgb_array', continuous=True)
        self.action_space = self.real_env.action_space

        # --- DYNAMIC OBSERVATION SPACE ---
        if self.mode == 'visual':
            # Dictionary: Image + Vector
            self.observation_space = gym.spaces.Dict({
                "image": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
                "vector": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(HIDDEN_DIM,), dtype=np.float32)
            })
        else:
            # Box: Latent + Vector
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(LATENT_DIM + HIDDEN_DIM,), dtype=np.float32
            )

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

    def _get_obs(self, z, h):
        h_vec = h.squeeze(0).squeeze(0).cpu().numpy()

        if self.mode == 'visual':
            with torch.no_grad():
                recon = self.vae.decode(z)
                recon_img = (recon.squeeze(0) * 255).clamp(0, 255).byte().cpu().numpy()
            return {
                "image": recon_img,
                "vector": h_vec
            }
        else:
            # Flatten z and concatenate with h
            z_vec = z.squeeze(0).cpu().numpy()
            return np.concatenate([z_vec, h_vec], axis=0)

    def reset(self, seed=None, options=None):
        obs, info = self.real_env.reset(seed=seed, options=options)

        warmup_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(WARMUP_STEPS):
            obs, _, _, _, _ = self.real_env.step(warmup_action)

        self.current_z = self._process_frame(obs)
        self.hidden_state = None
        h_zeros = torch.zeros(1, 1, HIDDEN_DIM).to(self.device)

        return self._get_obs(self.current_z, h_zeros), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.real_env.step(action)

        next_z = self._process_frame(obs)

        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device).view(1, 1, ACTION_DIM)
        z_tensor = self.current_z.view(1, 1, LATENT_DIM)

        with torch.no_grad():
            _, self.hidden_state = self.rnn(z_tensor, action_tensor, self.hidden_state)

        h_out = self.hidden_state[0]
        obs_ret = self._get_obs(next_z, h_out)
        self.current_z = next_z

        return obs_ret, reward, terminated, truncated, info

    def close(self):
        self.real_env.close()


def run_experiment(args):
    experiment_cfg = WEIGHT_PATHS.get(args.experiment_type)
    if not experiment_cfg:
        print(f"Error: Unknown experiment type '{args.experiment_type}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING EXPERIMENT: {args.experiment_type} ---")
    print(f"Device: {device}")
    print(f"Total Runs: {args.n_runs}")
    print(f"Timesteps per run: {args.total_timesteps}")

    base_log_dir = os.path.join(args.output_dir, args.experiment_type)
    os.makedirs(base_log_dir, exist_ok=True)

    for run_idx in range(1, args.n_runs + 1):
        print(f"\n> Starting Run {run_idx}/{args.n_runs}...")

        # 1. Setup specific log folder for this run
        run_dir = os.path.join(base_log_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        # 2. Setup Environment
        env = FlexibleWorldModelEnv(
            experiment_cfg['vae'],
            experiment_cfg['rnn'],
            experiment_cfg['mode'],
            device
        )

        # 3. Wrap env with Monitor to save CSV logs (Crucial for averaging later)
        env = Monitor(env, filename=os.path.join(run_dir, "monitor.csv"))

        # 4. Determine Policy Type
        if experiment_cfg['mode'] == 'visual':
            policy_type = "MultiInputPolicy"  # CNN for image + MLP for vector
        else:
            policy_type = "MlpPolicy"  # Standard MLP for vectors

        # 5. Initialize PPO
        model = PPO(
            policy_type,
            env,
            verbose=0,  # Keep it quiet to not flood terminal
            learning_rate=3e-4,
            ent_coef=0.01,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device=device,
            tensorboard_log=None  # We use Monitor CSVs instead
        )

        # 6. Train
        model.learn(total_timesteps=args.total_timesteps)

        # 7. Save Final Model
        model.save(os.path.join(run_dir, "final_model"))
        print(f"Run {run_idx} complete. Logs saved to {run_dir}")

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, required=True,
                        choices=['normal_latent', 'decoded_latent', 'lpips_latent'],
                        help='Which pipeline to train')
    parser.add_argument('--n_runs', type=int, default=10, help='How many times to repeat the training')
    parser.add_argument('--total_timesteps', type=int, default=500000, help='Steps per run')
    parser.add_argument('--output_dir', type=str, default='final_results', help='Where to save data')

    args = parser.parse_args()
    run_experiment(args)