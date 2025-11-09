import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio.v3 as iio
import torch
import torchvision.transforms as T
from PIL import Image
from vae_model import VAE, LATENT_DIM


def plot_training_logs(csv_path, save_dir='vae_logs'):
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Total Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Total Loss')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Total Loss')
    ax1.set_ylabel('Total Loss')
    ax1.legend()
    ax1.set_title('VAE Training: Total Loss')
    ax1.grid(True)

    # Plot 2: Reconstruction Loss (The "Accuracy" equivalent)
    ax2.plot(df['epoch'], df['train_recon_loss'], label='Train Recon Loss')
    ax2.plot(df['epoch'], df['val_recon_loss'], label='Validation Recon Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Loss')
    ax2.legend()
    ax2.set_title('VAE Training: Reconstruction Loss')
    ax2.grid(True)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_plots.png')
    fig.savefig(save_path)
    print(f"Training plots saved to {save_path}")
    plt.close(fig)


def create_vae_comparison_video(rollout_path, vae_model_path, output_video_path, fps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(z_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(vae_model_path, map_location=device))
    model.eval()

    with np.load(rollout_path) as data:
        observations = data['observations']

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
    ])

    reconstructed_frames = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(observations), batch_size):
            batch_np = observations[i:i + batch_size]

            batch_tensors = [transform(frame) for frame in batch_np]
            batch_in = torch.stack(batch_tensors).to(device)

            recon_batch, _, _ = model(batch_in)

            recon_batch_cpu = recon_batch.cpu()
            for tensor in recon_batch_cpu:
                frame_np = tensor.permute(1, 2, 0).numpy()
                frame_uint8 = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
                reconstructed_frames.append(frame_uint8)

    recon_array = np.stack(reconstructed_frames, axis=0)

    side_by_side_frames = np.concatenate((observations, recon_array), axis=2)

    print(f"Creating side-by-side video from {len(observations)} frames...")

    iio.imwrite(
        output_video_path,
        side_by_side_frames,
        fps=fps,
        pixelformat='yuv420p'
    )

    print(f"Successfully saved comparison video to {output_video_path}")


def plot_rnn_logs(csv_path, save_dir='rnn_logs'):
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MDN Loss')
    ax1.legend()
    ax1.set_title('MDN-RNN Training Loss')
    ax1.grid(True)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'rnn_training_plots.png')
    fig.savefig(save_path)
    print(f"Training plots saved to {save_path}")
    plt.close(fig)