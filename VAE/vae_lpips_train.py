import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import pandas as pd
import lpips
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VAE.vae_model import VAE, LATENT_DIM
from VAE.vae_dataset import VAE_Dataset

# Setup LPIPS
lpips_model = lpips.LPIPS(net='vgg').cuda()
for param in lpips_model.parameters():
    param.requires_grad = False


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    x_scaled = (x * 2) - 1
    recon_scaled = (recon_x * 2) - 1
    p_loss = lpips_model(recon_scaled, x_scaled).sum() * 1000.0

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + p_loss + kld
    return total_loss / x.size(0), (recon_loss + p_loss) / x.size(0), kld / x.size(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE for World Models')

    # Can now be a directory or a file
    parser.add_argument('--data_path', type=str, default='data/rollouts_v2_official', help='Directory containing .npy files')

    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Latent dimension size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default='vae_logs_lpips', help='Dir to save logs/models')
    parser.add_argument('--ckpt_dir', type=str, default='vae_checkpoints_lpips', help='Dir to save model checkpoints')
    parser.add_argument('--save_model', type=str, default='vae_lpips.pth', help='Final trained model filename')
    parser.add_argument('--save_interval', type=int, default=1, help='Save sample images every N epochs')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lpips_model = lpips_model.to(device)

    train_dataset = VAE_Dataset(args.data_path, train=True)
    val_dataset = VAE_Dataset(args.data_path, train=False)

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
    os.makedirs(args.ckpt_dir, exist_ok=True)

    training_logs = []

    print(f"Start training VAE (with LPIPS)... (Latent Dim: {args.latent_dim})")

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