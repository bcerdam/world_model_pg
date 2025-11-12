import torch
import torchvision.transforms as T
import numpy as np
import os
import argparse
from tqdm import tqdm
import imageio.v3 as iio
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from rnn_model import MDN_RNN, ACTION_DIM, HIDDEN_DIM


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

    vae = VAE(z_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(args.vae_model, map_location=device))
    vae.eval()

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

    print("Generating dreamed frames...")
    for t in tqdm(range(len(observations) - 1)):
        obs_t_np = observations[t]
        obs_tplus1_np = observations[t + 1]
        action_t_np = actions[t]

        obs_t_tensor = transform(obs_t_np).unsqueeze(0).to(device)

        with torch.no_grad():
            mu_t, logvar_t = vae.encode(obs_t_tensor)
            z_t = vae.reparameterize(mu_t, logvar_t)

            z_t_seq = z_t.unsqueeze(1)
            action_t_seq = torch.from_numpy(action_t_np).float().to(device).unsqueeze(0).unsqueeze(1)

            (log_pi, mu, log_sigma), hidden_state = rnn(z_t_seq, action_t_seq, hidden_state)

            predicted_z_tplus1 = get_most_likely_z(log_pi, mu, log_sigma)

            predicted_obs_tensor = vae.decode(predicted_z_tplus1)

        ground_truth_img = (T.ToTensor()(obs_tplus1_np) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        predicted_img = tensor_to_np(predicted_obs_tensor)

        comparison_frame = np.concatenate((ground_truth_img, predicted_img), axis=1)
        video_frames.append(comparison_frame)

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

    parser.add_argument('--rollout_path', type=str, default='data/vae_dataset/rollout_0.npz',
                        help='Path to a single .npz rollout')
    parser.add_argument('--vae_model', type=str, default='cluster_results/model_checkpoints/vae_checkpoints/vae_epoch_10.pth', help='Path to trained VAE model')
    parser.add_argument('--rnn_model', type=str, default='cluster_results/model_checkpoints/rnn_checkpoints/rnn_epoch_10.pth', help='Path to trained RNN model')
    parser.add_argument('--output_path', type=str, default='video_output/rnn_epoch_10.mp4',
                        help='Path to save the output .mp4 video')

    args = parser.parse_args()
    run_visualization(args)