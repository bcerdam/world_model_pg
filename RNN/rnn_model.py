import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_GAUSSIANS = 5
LATENT_DIM = 32
ACTION_DIM = 3
HIDDEN_DIM = 256


class MDN_RNN(nn.Module):
    def __init__(self, z_dim=LATENT_DIM, a_dim=ACTION_DIM, h_dim=HIDDEN_DIM, n_gaussians=N_GAUSSIANS):
        super(MDN_RNN, self).__init__()

        self.z_dim = z_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(z_dim + a_dim, h_dim, batch_first=True)

        self.mdn_head = nn.Linear(h_dim, (2 * z_dim + 1) * n_gaussians)

    def get_mixture_params(self, lstm_output):
        batch_size, seq_len, _ = lstm_output.shape

        mdn_params = self.mdn_head(lstm_output)

        mdn_params = mdn_params.view(
            batch_size, seq_len, self.n_gaussians, (2 * self.z_dim + 1)
        )

        log_pi = F.log_softmax(mdn_params[..., 0:1], dim=2)
        mu = mdn_params[..., 1: 1 + self.z_dim]
        log_sigma = mdn_params[..., 1 + self.z_dim:]

        return log_pi, mu, log_sigma

    def forward(self, z, a, h_in=None):
        lstm_in = torch.cat([z, a], dim=-1)

        if h_in is None:
            lstm_out, hidden_state = self.lstm(lstm_in)
        else:
            lstm_out, hidden_state = self.lstm(lstm_in, h_in)

        log_pi, mu, log_sigma = self.get_mixture_params(lstm_out)

        return (log_pi, mu, log_sigma), hidden_state


def mdn_loss(log_pi, mu, log_sigma, z_next):
    seq_len = z_next.size(1)

    z_next = z_next.unsqueeze(2).expand_as(mu)

    log_sigma = torch.clamp(log_sigma, -10.0, 10.0)
    sigma = torch.exp(log_sigma)

    log_prob_n = -0.5 * (
        torch.sum(
            ((z_next - mu) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi),
            dim=-1
        )
    )

    log_prob_weighted = log_pi.squeeze(-1) + log_prob_n

    log_likelihood = torch.logsumexp(log_prob_weighted, dim=2)

    total_loss = -torch.mean(log_likelihood)

    return total_loss