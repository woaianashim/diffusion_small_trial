import torch
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_dim = cfg.in_dim
        self.out_dim = cfg.out_dim

        layers = []
        # first layer
        layers.append(
            nn.Linear(
                self.in_dim + (cfg.condition_branch + 1) * cfg.condition_depth,
                cfg.hidden_dim,
            )
        )
        layers.append(nn.SiLU())

        # hidden layers
        for _ in range(cfg.num_layers - 1):
            layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            layers.append(nn.SiLU())

        # final layer: outputs D_out + 1 features
        layers.append(nn.Linear(cfg.hidden_dim, self.out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, noised, t, label=None):
        if label is None:
            label = torch.zeros(
                (
                    t.shape[0],
                    (self.cfg.condition_branch + 1) * self.cfg.condition_depth,
                ),
                device=t.device,
            )
        x = torch.cat([noised, t[..., None], label], dim=-1)
        return self.mlp(x)
