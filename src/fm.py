import torch
from .base_density_model import BaseDensityModel


class FlowMatching(BaseDensityModel):
    def precompute_sample_metadata(self, t_grid):
        self.dt = t_grid[1:] - t_grid[:-1]

    def step_along_vector(self, x, i, vector):
        return x + self.dt[i - 1] * vector

    def noise_forward(self, x):
        n = x.shape[0]
        t = torch.rand((n,), device=x.device)
        noise = torch.randn((n, 2), device=x.device)
        noised = noise * (t[:, None]) + x * (1 - t[:, None])
        u_target = x - noise
        return noised, u_target, t
