import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import pandas as pd

# These imports will work as long as rnn_model.py and rnn_dataset.py
# are in the same 'RNN/' folder as this script.
from rnn_model import MDN_RNN, mdn_loss, LATENT_DIM, ACTION_DIM, HIDDEN_DIM
from rnn_dataset import RNN_Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MDN-RNN for World Models')

    # --- PATHS UPDATED ---
    parser.add_argument('--data_path', type=str, default='data/rnn_dataset/rnn_dataset.npz',
                        help='Path to preprocessed rnn_dataset.npz')
    parser.add_argument('--log_dir', type=str, default='model_logs/rnn_logs',
                        help='Dir to save logs/models')
    parser.add_argument('--ckpt_dir', type=str, default='model_checkpoints/rnn_checkpoints',
                        help='Dir to save model checkpoints')
    # --- END OF PATH UPDATES ---

    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_model', type=str, default='rnn_final.pth', help='Final trained model filename')
    parser.add_argument('--save_interval', type=int, default=2, help='Save model weights every N epochs')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = RNN_Dataset(args.data_path, seq_len=args.seq_len, train=True)
    val_dataset = RNN_Dataset(args.data_path, seq_len=args.seq_len, train=False)

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

    model = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    training_logs = []

    print("Start training MDN-RNN...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader)

        for batch_idx, (z_t, a_t, z_next) in enumerate(pbar):
            z_t, a_t, z_next = z_t.to(device), a_t.to(device), z_next.to(device)
            optimizer.zero_grad()

            (log_pi, mu, log_sigma), _ = model(z_t, a_t)
            loss = mdn_loss(log_pi, mu, log_sigma, z_next)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} [Train Loss: {loss.item():.4f}]")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for z_t, a_t, z_next in val_loader:
                z_t, a_t, z_next = z_t.to(device), a_t.to(device), z_next.to(device)
                (log_pi, mu, log_sigma), _ = model(z_t, a_t)
                loss = mdn_loss(log_pi, mu, log_sigma, z_next)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"====> Epoch: {epoch} Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        training_logs.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'rnn_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    save_model_path = os.path.join(args.log_dir, args.save_model)
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    log_df = pd.DataFrame(training_logs)
    log_csv_path = os.path.join(args.log_dir, 'rnn_training_logs.csv')
    log_df.to_csv(log_csv_path, index=False)
    print(f"Training logs saved to {log_csv_path}")