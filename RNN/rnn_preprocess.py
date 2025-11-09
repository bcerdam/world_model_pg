import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
import sys

# --- PATH UPDATED ---
# Add project root to sys.path to allow importing from VAE module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
# --- END OF PATH UPDATE ---

BATCH_SIZE = 128


def preprocess_rollouts(data_dir, vae_model_path, output_path, max_rollouts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VAE(z_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(vae_model_path, map_location=device))
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])

    all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))[:max_rollouts]
    print(f"Found {len(all_files)} rollouts to process from {data_dir}...")

    all_latent_sequences = []
    all_action_sequences = []

    for f in tqdm(all_files):
        with np.load(f) as data:
            observations = data['observations']
            actions = data['actions']

        latent_sequence = []

        with torch.no_grad():
            for i in range(0, len(observations), BATCH_SIZE):
                obs_batch_np = observations[i:i + BATCH_SIZE]

                obs_batch_tensors = [transform(frame) for frame in obs_batch_np]
                obs_batch_in = torch.stack(obs_batch_tensors).to(device)

                mu, logvar = model.encode(obs_batch_in)
                z = model.reparameterize(mu, logvar)

                latent_sequence.append(z.cpu().numpy())

        latent_sequence = np.concatenate(latent_sequence, axis=0)

        all_latent_sequences.append(latent_sequence)
        all_action_sequences.append(actions)

    # --- PATH UPDATED ---
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # --- END OF PATH UPDATE ---

    np.savez_compressed(
        output_path,
        latents=np.array(all_latent_sequences, dtype=object),
        actions=np.array(all_action_sequences, dtype=object)
    )
    print(f"Preprocessed latent/action data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess dataset for RNN')

    # --- PATHS UPDATED ---
    parser.add_argument('--data_dir', type=str, default='data/vae_dataset',
                        help='Directory of .npz rollouts')
    parser.add_argument('--vae_model', type=str, default='model_checkpoints/vae_checkpoints/vae_epoch_10.pth',
                        help='Path to trained VAE model')
    parser.add_argument('--output_path', type=str, default='data/rnn_dataset/rnn_dataset.npz',
                        help='Path to save the new preprocessed file')
    # --- END OF PATH UPDATES ---

    parser.add_argument('--max_rollouts', type=int, default=100, help='Max rollouts to process')

    args = parser.parse_args()

    preprocess_rollouts(args.data_dir, args.vae_model, args.output_path, args.max_rollouts)