import torch
import torchvision.transforms as T
import numpy as np
import os
import argparse
from tqdm import tqdm
import imageio.v3 as iio
from PIL import Image
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from VAE.vae_model import VAE, LATENT_DIM
from RNN.rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM


def tensor_to_np(tensor):
    frame_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frame_uint8 = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
    return frame_uint8


def get_most_likely_z(log_pi, mu, log_sigma):
    best_gaussian_idx = torch.argmax(log_pi, dim=2)

    mu = mu.squeeze(0)
    best_mu = mu[torch.arange(mu.size(0)), best_gaussian_idx.squeeze()]

    return best_mu


def run_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE
    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(args.vae_model, map_location=device))
    vae.eval()

    # Load RNN
    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    rnn.load_state_dict(torch.load(args.rnn_model, map_location=device))
    rnn.eval()

    print(f"Loading rollout: {args.rollout_path}")
    with np.load(args.rollout_path) as data:
        observations = data['observations']
        actions = data['actions']

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])

    video_frames = []
    hidden_state = None

    print("Generating one-step predictions...")
    # We go up to len-1 because we predict t+1 from t
    for t in tqdm(range(len(observations) - 1)):
        obs_t_np = observations[t]
        obs_tplus1_np = observations[t + 1]
        action_t_np = actions[t]

        obs_t_tensor = transform(obs_t_np).unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. Encode current frame
            mu_t, logvar_t = vae.encode(obs_t_tensor)
            z_t = vae.reparameterize(mu_t, logvar_t)

            # Prepare inputs for RNN
            z_t_seq = z_t.unsqueeze(1)
            action_t_seq = torch.from_numpy(action_t_np).float().to(device).unsqueeze(0).unsqueeze(1)

            # 2. RNN Predicts next z
            # FIX: Unpack 5 values (pi, mu, sigma, reward, done) and ignore the last two
            (log_pi, mu, log_sigma, _, _), hidden_state = rnn(z_t_seq, action_t_seq, hidden_state)

            # 3. Get most likely z and decode it
            predicted_z_tplus1 = get_most_likely_z(log_pi, mu, log_sigma)
            predicted_obs_tensor = vae.decode(predicted_z_tplus1)

        # 4. Prepare comparison image
        ground_truth_img = (T.ToTensor()(obs_tplus1_np) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        predicted_img = tensor_to_np(predicted_obs_tensor)

        comparison_frame = np.concatenate((ground_truth_img, predicted_img), axis=1)
        video_frames.append(comparison_frame)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"Saving comparison video to {args.output_path}...")
    iio.imwrite(
        args.output_path,
        video_frames,
        fps=30,
        pixelformat='yuv420p'
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize RNN one-step predictions')

    parser.add_argument('--rollout_path', type=str, default='data/rollouts_v2/rollout_20.npz',
                        help='Path to a single .npz rollout')
    parser.add_argument('--vae_model', type=str, default='model_checkpoints/vae_2_checkpoints/vae_epoch_19.pth',
                        help='Path to trained VAE model')
    parser.add_argument('--rnn_model', type=str, default='rnn_2_checkpoints/rnn_epoch_1.pth',
                        help='Path to trained RNN model')
    parser.add_argument('--output_path', type=str, default='video_output/test_rnn.mp4',
                        help='Path to save the output .mp4 video')

    args = parser.parse_args()
    run_visualization(args)