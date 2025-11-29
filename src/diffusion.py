import torch
from tqdm import tqdm
from .base_density_model import BaseDensityModel


class Diffusion(BaseDensityModel):
    def noise_forward(self, x):
        n = x.shape[0]
        t = torch.rand((n,), device=x.device)
        alpha_bar = self._alpha_bar(t).to(x.device)[:, None]
        noise = torch.randn((n, 2), device=x.device)
        noised = x * alpha_bar.sqrt() + noise * (1 - alpha_bar).sqrt()
        target_vector = noise
        return noised, target_vector, t

    def precompute_sample_metadata(self, t_grid):
        self.alpha_bar = self._alpha_bar(t_grid)
        alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.alpha = torch.cat([torch.ones(1, device=self.cfg.device), alpha])
        self.beta = 1.0 - self.alpha

    def step_along_vector(self, x, i, vector):
        if self.cfg.sampler.mode == "DDIM":
            mean = (
                (x - (1 - self.alpha_bar[i]).sqrt() * vector)
                / self.alpha_bar[i].sqrt()
                * self.alpha_bar[i - 1].sqrt()
            )
        else:
            mean = (
                x - (self.beta[i] / torch.sqrt(1.0 - self.alpha_bar[i])) * vector
            ) / torch.sqrt(self.alpha[i])

        if i > 1 and self.alpha[i - 1] < 1 - 1e-6:
            eps = torch.randn_like(x)
            if self.cfg.sampler.mode == "DDIM":
                sigma_eta = (
                    self.cfg.sampler.eta
                    * ((1 - self.alpha_bar[i - 1]) / (1 - self.alpha_bar[i])).sqrt()
                    * (1 - self.alpha_bar[i] / self.alpha_bar[i - 1]).sqrt()
                )
                x = (
                    mean
                    + (1 - self.alpha_bar[i - 1] - sigma_eta**2).sqrt() * vector
                    + sigma_eta * eps
                )
            else:
                x = mean + self.beta[i].sqrt() * eps
        else:
            x = mean
        return x

    def _alpha_bar(self, t):
        s = self.cfg.sampler.s
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        return alpha_bar

    def get_gt_vf(self, noised, t, labels=None):
        alpha_bar = self._alpha_bar(t).to(self.cfg.device)
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

            noise = (
                ((noised_chunk * alpha_bar.rsqrt())[None] - pts[:, None])
                * alpha_bar.sqrt()
                * (1 - alpha_bar).rsqrt()
            )
            dist2 = (noise**2).sum(-1, keepdim=True)
            dist2 -= dist2.min(0, keepdim=True).values
            weight = torch.exp(-dist2 / 2)
            diff_weighted = noise * weight
            E = diff_weighted.sum(dim=0) / (weight.sum(0) + 1e-9)
            res.append(E)
        vf = torch.cat(res, dim=0)
        return vf
