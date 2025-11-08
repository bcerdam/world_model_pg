import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import pandas as pd

from vae_model import VAE, LATENT_DIM
from vae_dataset import VAE_Dataset


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kld

    return total_loss / x.size(0), recon_loss / x.size(0), kld / x.size(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE for World Models')

    parser.add_argument('--data_dir', type=str, default='dataset', help='Directory of dataset')
    parser.add_argument('--max_rollouts', type=int, default=2000, help='Max rollouts to load')
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Latent dimension size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='vae_logs', help='Dir to save logs/models')

    # WHERE THE CHANGE IS: This new argument defines the separate checkpoint folder.
    parser.add_argument('--ckpt_dir', type=str, default='vae_checkpoints', help='Dir to save model checkpoints')

    parser.add_argument('--save_model', type=str, default='vae_final.pth', help='Final trained model filename')
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save sample images and model weights every N epochs')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = VAE_Dataset(args.data_dir, train=True, max_rollouts=args.max_rollouts)
    val_dataset = VAE_Dataset(args.data_dir, train=False, max_rollouts=args.max_rollouts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = VAE(z_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.log_dir, exist_ok=True)

    # WHERE THE CHANGE IS: We create the new, separate folder for checkpoints.
    os.makedirs(args.ckpt_dir, exist_ok=True)

    training_logs = []

    print(f"Start training VAE... (Latent Dim: {args.latent_dim})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_recon, train_kld = 0, 0, 0
        pbar = tqdm(train_loader)

        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)
            loss, recon_b, kld_b = vae_loss(recon, data, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_b.item()
            train_kld += kld_b.item()
            pbar.set_description(f"Epoch {epoch} [Train Loss: {loss.item():.4f}]")

        model.eval()
        val_loss, val_recon, val_kld = 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, recon_b, kld_b = vae_loss(recon, data, mu, logvar)

                val_loss += loss.item()
                val_recon += recon_b.item()
                val_kld += kld_b.item()

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kld /= len(train_loader)
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kld /= len(val_loader)

        print(f"====> Epoch: {epoch} Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        training_logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_recon_loss': train_recon,
            'val_recon_loss': val_recon,
            'train_kld': train_kld,
            'val_kld': val_kld
        })

        if epoch % args.save_interval == 0:
            with torch.no_grad():
                sample_data = next(iter(val_loader)).to(device)
                sample_recon, _, _ = model(sample_data)

                comparison = torch.cat([
                    sample_data[:8].view(-1, 3, 64, 64),
                    sample_recon[:8].view(-1, 3, 64, 64)
                ])

                save_path = os.path.join(args.log_dir, f'recon_epoch_{epoch}.png')
                save_image(comparison.cpu(), save_path, nrow=8)

            # WHERE THE CHANGE IS: We save a model checkpoint to the new directory.
            ckpt_path = os.path.join(args.ckpt_dir, f'vae_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    save_model_path = os.path.join(args.log_dir, args.save_model)
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    log_df = pd.DataFrame(training_logs)
    log_csv_path = os.path.join(args.log_dir, 'training_logs.csv')
    log_df.to_csv(log_csv_path, index=False)
    print(f"Training logs saved to {log_csv_path}")