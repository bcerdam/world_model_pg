import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM
from VAE.vae_model import LATENT_DIM

ENV_NAME = 'CarRacing-v3'


def sample_mdn(log_pi, mu, log_sigma, temperature=1.0):
    log_pi = log_pi / temperature
    sigma = torch.exp(log_sigma) * np.sqrt(temperature)

    pi_probs = F.softmax(log_pi.squeeze(-1), dim=-1)
    pi_probs = pi_probs.squeeze(1)

    cat = torch.distributions.Categorical(pi_probs)
    gaussian_idx = cat.sample()

    z_samples = []
    for b in range(mu.size(0)):
        idx = gaussian_idx[b]
        selected_mu = mu[b, 0, idx, :]
        selected_sigma = sigma[b, 0, idx, :]
        epsilon = torch.randn_like(selected_mu)
        z = selected_mu + epsilon * selected_sigma
        z_samples.append(z)

    return torch.stack(z_samples).unsqueeze(1)


class DreamEnv(gym.Env):
    def __init__(self, rnn_path, data_path, device, temperature=1.15):
        super(DreamEnv, self).__init__()

        self.device = device
        self.temperature = temperature

        self.rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(self.device)
        self.rnn.load_state_dict(torch.load(rnn_path, map_location=self.device))
        self.rnn.eval()

        print(f"Loading dataset from {data_path}...")
        with np.load(data_path, allow_pickle=True) as data:
            self.all_mus = data['mus']
            self.all_logvars = data['logvars']

        self.num_rollouts = len(self.all_mus)
        print(f"Loaded {self.num_rollouts} rollouts for seeding.")

        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]),
                                           dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(LATENT_DIM + HIDDEN_DIM,), dtype=np.float32
        )

        self.current_z = None
        self.hidden_state = None
        self.time_step = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rollout_idx = np.random.randint(0, self.num_rollouts)
        rollout_length = len(self.all_mus[rollout_idx])
        time_idx = np.random.randint(0, rollout_length)

        mu_val = self.all_mus[rollout_idx][time_idx]
        logvar_val = self.all_logvars[rollout_idx][time_idx]

        # mu = torch.from_numpy(mu_val).to(self.device)
        # logvar = torch.from_numpy(logvar_val).to(self.device)
        mu = torch.from_numpy(mu_val.astype(np.float32)).to(self.device)
        logvar = torch.from_numpy(logvar_val.astype(np.float32)).to(self.device)


        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_0 = mu + eps * std

        self.current_z = z_0.view(1, 1, LATENT_DIM)

        self.hidden_state = None
        h_zeros = torch.zeros(1, HIDDEN_DIM).to(self.device)

        self.time_step = 0

        state = torch.cat([self.current_z.squeeze(0).squeeze(0), h_zeros.squeeze(0)], dim=0).cpu().numpy()
        return state, {}

    def step(self, action):
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device).view(1, 1, ACTION_DIM)

        with torch.no_grad():
            (log_pi, mu, log_sigma, pred_r, pred_d_logits), self.hidden_state = self.rnn(
                self.current_z, action_tensor, self.hidden_state
            )

            next_z = sample_mdn(log_pi, mu, log_sigma, temperature=self.temperature)

            reward = pred_r.item()
            done_logit = pred_d_logits.item()

        self.current_z = next_z
        self.time_step += 1

        done_prob = 1 / (1 + np.exp(-done_logit))

        terminated = done_prob > 0.5
        truncated = self.time_step >= self.max_steps
        done = terminated or truncated

        h_t = self.hidden_state[0].squeeze(0).squeeze(0)
        next_state = torch.cat([self.current_z.squeeze(0).squeeze(0), h_t], dim=0).cpu().numpy()

        return next_state, reward, terminated, truncated, {}

    def close(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_path', type=str,
                        default='rnn_checkpoints/rnn_epoch_4.pth')
    parser.add_argument('--data_path', type=str, default='data/rnn_dataset/rnn_dataset.npz')
    parser.add_argument('--save_path', type=str, default='model_checkpoints/controller_2_checkpoints/ppo_dream')
    parser.add_argument('--total_timesteps', type=int, default=3000000)
    parser.add_argument('--temperature', type=float, default=1.15, help='Dream temperature (higher = harder)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PPO in DREAM on device: {device}")

    env = DreamEnv(args.rnn_path, args.data_path, device, temperature=args.temperature)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./ppo_dream_checkpoints/',
        name_prefix='ppo_dream'
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.01,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        device=device
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}.zip")