import torch
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    def __init__(self, z_dim, h_dim, a_dim):
        super(Controller, self).__init__()
        self.fc = nn.Linear(z_dim + h_dim, a_dim)

    def forward(self, z, h):
        combined = torch.cat([z, h], dim=1)
        action = self.fc(combined)
        return torch.tanh(action)

    def get_flat_params(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            num_param = p.numel()
            p.data.copy_(torch.from_numpy(flat_params[idx:idx+num_param]).view_as(p))
            idx += num_param