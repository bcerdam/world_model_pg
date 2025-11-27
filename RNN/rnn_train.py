import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RNN.rnn_model import MDN_RNN, mdn_loss, LATENT_DIM, ACTION_DIM, HIDDEN_DIM
from RNN.rnn_dataset import RNN_Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MDN-RNN for World Models')

    parser.add_argument('--data_path', type=str, default='data/rnn_2_dataset/rnn_dataset.npz',
                        help='Path to preprocessed rnn_dataset.npz')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='RNN hidden dimension')

    # Removed --dropout and --weight_decay arguments

    parser.add_argument('--log_dir', type=str, default='rnn_2_logs', help='Dir to save logs/models')
    parser.add_argument('--ckpt_dir', type=str, default='rnn_2_checkpoints', help='Dir to save model checkpoints')
    parser.add_argument('--save_model', type=str, default='rnn_final.pth', help='Final trained model filename')
    parser.add_argument('--save_interval', type=int, default=1, help='Save model weights every N epochs')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = RNN_Dataset(args.data_path, seq_len=args.seq_len, train=True)
    val_dataset = RNN_Dataset(args.data_path, seq_len=args.seq_len, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model without dropout
    model = MDN_RNN(LATENT_DIM, ACTION_DIM, args.hidden_dim).to(device)

    # Optimizer without weight_decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    training_logs = []

    print("Start training MDN-RNN...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_mdn, train_rew, train_done = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (z_t, a_t, z_next, r_next, d_next) in enumerate(pbar):
            z_t, a_t, z_next = z_t.to(device), a_t.to(device), z_next.to(device)
            r_next, d_next = r_next.to(device), d_next.to(device)

            optimizer.zero_grad()

            (log_pi, mu, log_sigma, pred_r, pred_d_logits), _ = model(z_t, a_t)

            l_mdn = mdn_loss(log_pi, mu, log_sigma, z_next)
            l_reward = F.mse_loss(pred_r.squeeze(-1), r_next)
            l_done = F.binary_cross_entropy_with_logits(pred_d_logits.squeeze(-1), d_next)

            # loss = l_mdn + l_reward + l_done
            loss = l_mdn + (20.0 * l_reward) + (20.0 * l_done)
            # loss = l_mdn

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mdn += l_mdn.item()
            train_rew += l_reward.item()
            train_done += l_done.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for z_t, a_t, z_next, r_next, d_next in val_loader:
                z_t, a_t, z_next = z_t.to(device), a_t.to(device), z_next.to(device)
                r_next, d_next = r_next.to(device), d_next.to(device)

                (log_pi, mu, log_sigma, pred_r, pred_d_logits), _ = model(z_t, a_t)

                l_mdn = mdn_loss(log_pi, mu, log_sigma, z_next)
                l_reward = F.mse_loss(pred_r.squeeze(-1), r_next)
                l_done = F.binary_cross_entropy_with_logits(pred_d_logits.squeeze(-1), d_next)

                # val_loss += (l_mdn + l_reward + l_done).item()
                val_loss += l_mdn + (20.0 * l_reward) + (20.0 * l_done)
                # val_loss += l_mdn.item()

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