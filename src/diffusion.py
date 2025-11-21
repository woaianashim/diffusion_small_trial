import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import Normal
from tqdm import tqdm
from src.model import DenseNet
from .data import FracTree
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
import logging


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.data = FracTree(cfg.data)
        self.model = DenseNet(cfg.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.optim)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.epochs,
            eta_min=1e-6,
        )
        self.loss = torch.nn.MSELoss(reduction="none")  # No reduction for logging
        self.to(cfg.device)

    def run_train(self):
        self.train()

        dataloader = DataLoader(self.data, **self.cfg.dataloader, collate_fn=None)
        logging.info("Starting training.")
        for ep in range(self.cfg.epochs):
            seen = 0
            n_bins = self.cfg.logging.n_loss_bins
            epoch_loss = torch.zeros((n_bins,), device=self.device)
            epoch_seen = torch.ones((n_bins,), device=self.device)
            for batch in tqdm(dataloader):
                seen += len(batch["gt"])
                x, label = (
                    batch["gt"].to(self.device).squeeze(0),
                    batch["label"].to(self.device).squeeze(0),
                )
                noised, noise, t = self.noise_forward(x)
                label_mask = torch.rand_like(label) > 0.5
                preds = self.model(noised, t, label * label_mask)
                losses = self.loss(preds, noise).mean(-1)
                t_bins = (t * n_bins).floor().to(int)
                epoch_loss.scatter_add_(0, t_bins, losses)
                epoch_seen.scatter_add_(0, t_bins, torch.ones_like(losses))

                self.optimizer.zero_grad()
                losses.mean().backward()
                self.optimizer.step()
            self.scheduler.step()
            avg_epoch_loss = (epoch_loss / epoch_seen).detach().cpu()
            logging.info(f"Average Epoch {ep} loss: {avg_epoch_loss}")
            log_path = Path(HydraConfig.get().run.dir)
            torch.save(avg_epoch_loss, os.path.join(log_path, f"losses_{ep}.pt"))
            if self.cfg.save_period > 0 and ep % self.cfg.save_period == 0:
                torch.save(self.state_dict(), "checkpoint.pt")

    def sample(self, n, labels=None):
        self.eval()
        device = self.device
        if labels is not None:
            labels = labels.to(self.device)

        timesteps = self.cfg.sampler.num_steps
        t_grid = (
            torch.linspace(0.0, 1.0, timesteps + 2, device=device)[:-1]
            ** self.cfg.sampler.rho
        )

        alpha_bar = self.alpha_bar(t_grid)
        alpha = alpha_bar[1:] / alpha_bar[:-1]
        print(alpha_bar.shape)
        alpha = torch.cat([torch.ones(1, device=self.cfg.device), alpha])
        beta = 1.0 - alpha

        x = torch.randn(n, 2, device=device)  # Noise
        steps = [x]

        with torch.no_grad():
            for i in range(timesteps, 0, -1):
                t = t_grid[i].expand(n)
                eps_theta_cond = self.model(x, t, labels)
                eps_theta_uncond = self.model(x, t)
                w = self.cfg.sampler.cfg_omega
                eps_theta = w * eps_theta_cond + (1 - w) * eps_theta_uncond

                if self.cfg.sampler.mode == "DDIM":
                    mean = (
                        (x - (1 - alpha_bar[i]).sqrt() * eps_theta)
                        / alpha_bar[i].sqrt()
                        * alpha_bar[i - 1].sqrt()
                    )
                else:
                    mean = (
                        x - (beta[i] / torch.sqrt(1.0 - alpha_bar[i])) * eps_theta
                    ) / torch.sqrt(alpha[i])

                if i > 1 and alpha[i - 1] < 1 - 1e-6:
                    eps = torch.randn_like(x)
                    if self.cfg.sampler.mode == "DDIM":
                        sigma_eta = (
                            self.cfg.sampler.eta
                            * ((1 - alpha_bar[i - 1]) / (1 - alpha_bar[i])).sqrt()
                            * (1 - alpha_bar[i] / alpha_bar[i - 1]).sqrt()
                        )
                        x = (
                            mean
                            + (1 - alpha_bar[i - 1] - sigma_eta**2).sqrt() * eps_theta
                            + sigma_eta * eps
                        )
                    else:
                        x = mean + beta[i].sqrt() * eps
                else:
                    x = mean
                steps.append(x)

        return x, steps

    def noise_forward(self, x):
        n = x.shape[0]
        t = torch.rand((n,), device=x.device)
        alpha_bar = self.alpha_bar(t).to(x.device)[:, None]
        noise = torch.randn((n, 2), device=x.device)
        noised = x * alpha_bar.sqrt() + noise * (1 - alpha_bar).sqrt()
        return noised, noise, t

    def alpha_bar(self, t):
        s = self.cfg.sampler.s
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        # alpha_bar = alpha_bar / alpha_bar[0]
        return alpha_bar
