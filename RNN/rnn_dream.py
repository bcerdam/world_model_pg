import torch
import torchvision.transforms as T
import numpy as np
import os
import argparse
from tqdm import tqdm
import imageio.v3 as iio
from PIL import Image
import torch.nn.functional as F
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM


# --- MDN Sampling Helper ---
def sample_mdn(log_pi, mu, log_sigma, temperature=1.0):
    """
    Sample z from the Mixture Density Network outputs.
    """
    # Adjust temperature
    log_pi = log_pi / temperature
    sigma = torch.exp(log_sigma) * np.sqrt(temperature)

    # 1. Pick which Gaussian to use
    # log_pi shape: (batch, seq, n_gaussians, 1)
    # We squeeze the last dim to get (batch, seq, n_gaussians)
    pi_probs = F.softmax(log_pi.squeeze(-1), dim=-1)

    # We execute only for seq len 1, so squeeze seq dim: (batch, n_gaussians)
    pi_probs = pi_probs.squeeze(1)

    # Create categorical distribution: shape (batch, n_gaussians)
    cat = torch.distributions.Categorical(pi_probs)
    gaussian_idx = cat.sample()  # shape: (batch,)

    # 2. Sample z from that specific Gaussian
    # mu shape: (batch, seq, n_gaussians, z_dim)
    z_samples = []
    for b in range(mu.size(0)):
        idx = gaussian_idx[b]
        # Select the specific gaussian for this batch item
        selected_mu = mu[b, 0, idx, :]  # (z_dim)
        selected_sigma = sigma[b, 0, idx, :]  # (z_dim)

        # Standard normal sample
        epsilon = torch.randn_like(selected_mu)
        z = selected_mu + epsilon * selected_sigma
        z_samples.append(z)

    return torch.stack(z_samples).unsqueeze(1)  # Return shape (batch, 1, z_dim)


def run_dream(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(args.vae_model, map_location=device))
    vae.eval()

    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    rnn.load_state_dict(torch.load(args.rnn_model, map_location=device))
    rnn.eval()

    # --- Initialize the "Dream" ---
    print(f"Seeding dream from: {args.rollout_path}")
    with np.load(args.rollout_path) as data:
        first_frame = data['observations'][0]

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])

    first_frame_tensor = transform(first_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        mu_0, logvar_0 = vae.encode(first_frame_tensor)
        current_z = mu_0.unsqueeze(1)  # Shape: (1, 1, 32)

    # --- Initialize Policy State ---
    current_action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    frames_remaining = 0
    driving_straight = True

    hidden_state = None
    dreamed_latents = []

    print(f"Dreaming for {args.dream_length} steps...")

    for _ in tqdm(range(args.dream_length)):

        # --- 1. Policy Step ---
        if frames_remaining <= 0:
            if driving_straight:
                current_action[0] = np.random.uniform(-1.0, 1.0)
                frames_remaining = np.random.randint(args.turn_min, args.turn_max)
                driving_straight = False
            else:
                current_action[0] = 0.0
                frames_remaining = np.random.randint(args.straight_min, args.straight_max)
                driving_straight = True

        frames_remaining -= 1
        current_action[1] = 1.0
        current_action[2] = 0.0

        action_tensor = torch.from_numpy(current_action).float().to(device)
        action_tensor = action_tensor.view(1, 1, ACTION_DIM)  # (1, 1, 3)

        # --- 2. RNN Step ---
        with torch.no_grad():
            # This line caused the error before because current_z was 4D
            (log_pi, mu, log_sigma), hidden_state = rnn(current_z, action_tensor, hidden_state)

            # Sample the next z
            next_z = sample_mdn(log_pi, mu, log_sigma, temperature=args.temperature)

            dreamed_latents.append(next_z.squeeze(1))
            current_z = next_z

    # --- Decode the Dream ---
    print("Decoding dreamed latents to video...")
    dreamed_frames = []
    batch_size = 32
    all_latents = torch.cat(dreamed_latents, dim=0)

    with torch.no_grad():
        for i in range(0, len(all_latents), batch_size):
            batch_z = all_latents[i:i + batch_size]
            recon_batch = vae.decode(batch_z)

            recon_batch_cpu = recon_batch.cpu()
            for tensor in recon_batch_cpu:
                frame_np = tensor.permute(1, 2, 0).numpy()
                frame_uint8 = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
                dreamed_frames.append(frame_uint8)

    print(f"Saving video to {args.output_path}...")
    iio.imwrite(
        args.output_path,
        dreamed_frames,
        fps=30,
        pixelformat='yuv420p'
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a dream rollout using the World Model')

    # Default paths updated to match your likely folder structure
    parser.add_argument('--rollout_path', type=str, default='data/vae_dataset/rollout_0.npz',
                        help='File to seed the dream')
    parser.add_argument('--vae_model', type=str, default='cluster_results/model_checkpoints/vae_checkpoints/vae_epoch_10.pth',
                        help='Path to trained VAE')
    parser.add_argument('--rnn_model', type=str, default='cluster_results/model_checkpoints/rnn_checkpoints/rnn_epoch_4.pth',
                        help='Path to trained RNN')
    parser.add_argument('--output_path', type=str, default='video_output/RNN/random_dream.mp4', help='Output video file')
    parser.add_argument('--dream_length', type=int, default=1000, help='Number of frames to dream')
    parser.add_argument('--temperature', type=float, default=0.75, help='Sampling temperature')

    parser.add_argument('--turn_min', type=int, default=1)
    parser.add_argument('--turn_max', type=int, default=2)
    parser.add_argument('--straight_min', type=int, default=1)
    parser.add_argument('--straight_max', type=int, default=2)

    args = parser.parse_args()
    run_dream(args)