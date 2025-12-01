import torch
from tqdm import tqdm
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

    def get_gt_vf(self, noised, t, labels=None, custom_vf_fn=None):
        noised = noised.to(self.cfg.device)
        t = t.to(self.cfg.device)
        vf_fn = custom_vf_fn if custom_vf_fn else self.get_gt_vf_chunk
        batch_size = noised.shape[0]

        n_pts = 30000
        pts, _ = self.data.sample_points(n_pts)  # [N_pts, 2]
        pts = pts.to(self.cfg.device)

        # To avoid OOM
        chunk_size = 1024

        res = []
        brun = (
            (range(0, batch_size, chunk_size))
            if self.quiet
            else tqdm(range(0, batch_size, chunk_size))
        )
        for start in brun:
            end = min(start + chunk_size, batch_size)
            noised_chunk = noised[start:end]

            E = vf_fn(noised_chunk, pts, t)
            res.append(E)
        vf = torch.cat(res, dim=0)
        return vf

    def get_gt_vf_chunk(self, noised, data_points, t):
        noise = ((noised / (1 - t))[None] - data_points[:, None]) * (1 - t) / t
        dist2 = (noise**2).sum(-1, keepdim=True)
        dist2 -= dist2.min(0, keepdim=True).values
        weight = torch.exp(-dist2 / 2)
        diff_weighted = (data_points[:, None] - noise) * weight
        E = diff_weighted.sum(dim=0) / (weight.sum(0) + 1e-9)
        return E
