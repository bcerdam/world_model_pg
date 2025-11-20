import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from stable_baselines3 import PPO
import imageio.v3 as iio
import argparse
import os
import sys
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM


def sample_mdn(log_pi, mu, log_sigma, temperature=1.0):
    log_pi = log_pi / temperature
    sigma = torch.exp(log_sigma) * np.sqrt(temperature)
    pi_probs = F.softmax(log_pi.squeeze(-1), dim=-1).squeeze(1)
    cat = torch.distributions.Categorical(pi_probs)
    gaussian_idx = cat.sample()

    z_samples = []
    for b in range(mu.size(0)):
        idx = gaussian_idx[b]
        selected_mu = mu[b, 0, idx, :]
        selected_sigma = sigma[b, 0, idx, :]
        epsilon = torch.randn_like(selected_mu)
        z_samples.append(selected_mu + epsilon * selected_sigma)

    return torch.stack(z_samples).unsqueeze(1)


def visualize_dream(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.eval()

    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    rnn.load_state_dict(torch.load(args.rnn_path, map_location=device))
    rnn.eval()

    print(f"Loading PPO agent: {args.ppo_path}")
    model = PPO.load(args.ppo_path, device=device)

    print("Seeding dream...")
    with np.load(args.data_path, allow_pickle=True) as data:
        all_mus = data['mus']
        all_logvars = data['logvars']

    rollout_idx = np.random.randint(0, len(all_mus))
    time_idx = np.random.randint(0, 50)

    mu_val = all_mus[rollout_idx][time_idx]
    logvar_val = all_logvars[rollout_idx][time_idx]

    mu = torch.from_numpy(mu_val).to(device).unsqueeze(0)
    logvar = torch.from_numpy(logvar_val).to(device).unsqueeze(0)

    z = (mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)).view(1, 1, LATENT_DIM)

    frames = []
    hidden_state = None
    h_zeros = torch.zeros(1, HIDDEN_DIM).to(device)

    print(f"Generating {args.steps} dream steps...")

    for _ in range(args.steps):
        with torch.no_grad():
            recon = vae.decode(z)
            img = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(img)

        if hidden_state is None:
            h_t = h_zeros
        else:
            h_t = hidden_state[0].squeeze(0)

        obs_vector = torch.cat([z.squeeze(0), h_t], dim=1).cpu().numpy()
        action, _ = model.predict(obs_vector, deterministic=True)

        action_tensor = torch.tensor(action, dtype=torch.float32).to(device).view(1, 1, ACTION_DIM)

        with torch.no_grad():
            (log_pi, mu, log_sigma, pred_r, pred_d), hidden_state = rnn(z, action_tensor, hidden_state)
            z = sample_mdn(log_pi, mu, log_sigma, temperature=args.temperature)

    print(f"Saving video to {args.output_path}...")
    iio.imwrite(args.output_path, frames, fps=30, pixelformat='yuv420p')
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_path', type=str,
                        default='cluster_results/model_checkpoints/vae_checkpoints/vae_epoch_10.pth')
    parser.add_argument('--rnn_path', type=str,
                        default='cluster_results/model_checkpoints/rnn_checkpoints/rnn_epoch_4.pth')
    parser.add_argument('--ppo_path', type=str, default='ppo_checkpoints/ppo_model_300000_steps.zip', help='Path to .zip PPO model')
    parser.add_argument('--data_path', type=str, default='data/rnn_dataset/rnn_dataset.npz')
    parser.add_argument('--output_path', type=str, default='video_output/ppo_run_dream_1.mp4')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()
    visualize_dream(args)