import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
                seen += len(batch["t"])
                noised, t, label, noise = (
                    batch["noised"].to(self.device).squeeze(0),
                    batch["t"].to(self.device).squeeze(0),
                    batch["label"].to(self.device).squeeze(0),
                    batch["noise"].to(self.device).squeeze(0),
                )
                label_mask = torch.rand_like(label) > 0.5
                preds = self.model(noised, t, label * label_mask)
                losses = self.loss(preds, noise).mean(-1)
                t_bins = (t * n_bins).floor().to(int)
                epoch_loss.scatter_add_(0, t_bins, losses)
                epoch_seen.scatter_add_(0, t_bins, torch.ones_like(losses))

                self.optimizer.zero_grad()
                losses.mean().backward()
                self.optimizer.step()
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
        s = self.data.cfg.sampler.s
        t_grid = (torch.arange(timesteps, device=device) + 0.5) / timesteps

        alpha_bar = torch.cos((t_grid + s) / (1.0 + s) * torch.pi / 2.0) ** 2

        x = torch.randn(n, 2, device=device)  # Noise
        steps = [x]

        with torch.no_grad():
            for i in reversed(range(timesteps)):
                t = t_grid[i].expand(n)
                eps_theta_cond = self.model(x, t, labels)
                eps_theta_uncond = self.model(x, t)
                w = self.cfg.sampler.cfg_omega
                eps_theta = w * eps_theta_cond + (1 - w) * eps_theta_uncond

                alpha_bar_t = alpha_bar[i]
                sqrt_alpha_bar_t = alpha_bar_t.sqrt()
                sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()

                x0_pred = (
                    x - sqrt_one_minus_alpha_bar_t * eps_theta
                ) / sqrt_alpha_bar_t

                if i > 0:
                    alpha_bar_prev = alpha_bar[i - 1]
                    x = (
                        alpha_bar_prev.sqrt() * x0_pred
                        + (1.0 - alpha_bar_prev).sqrt() * eps_theta
                    )
                else:
                    x = x0_pred
                steps.append(x)

        return x, steps
