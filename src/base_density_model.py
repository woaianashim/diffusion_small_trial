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


class BaseDensityModel(nn.Module):
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
        self.loss = torch.nn.MSELoss(reduction="none")
        self.to(self.device)

    def load(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or f"checkpoints/{self.__class__.__name__}.pt"
        self.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device(self.cfg.device))
        )

    def run_train(self):
        self.train()

        dataloader = DataLoader(self.data, **self.cfg.dataloader, collate_fn=None)
        logging.info(f"Starting training {self.__class__.__name__}.")
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
                noised, target_vector, t = self.noise_forward(x)
                label_mask = torch.rand_like(label) > 0.5
                preds = self.model(noised, t, label * label_mask)
                losses = self.loss(preds, target_vector).mean(-1)  # (N,)
                t_bins = (t * n_bins).floor().to(torch.int64)
                epoch_loss.scatter_add_(0, t_bins, losses)
                epoch_seen.scatter_add_(
                    0, t_bins, torch.ones_like(losses, device=self.device)
                )
                self.optimizer.zero_grad()
                losses.mean().backward()
                self.optimizer.step()
            self.scheduler.step()
            avg_epoch_loss = (epoch_loss / epoch_seen).detach().cpu()
            logging.info(f"[FM] Average Epoch {ep} loss per bin: {avg_epoch_loss}")
            log_path = Path(HydraConfig.get().run.dir)
            torch.save(avg_epoch_loss, os.path.join(log_path, f"fm_losses_{ep}.pt"))
            if self.cfg.save_period > 0 and ep % self.cfg.save_period == 0:
                torch.save(
                    self.state_dict(), f"checkpoints/{self.__class__.__name__}.pt"
                )

    def noise_forward(self, x):
        raise NotImplementedError

    def sampling_times(self):
        timesteps = self.cfg.sampler.num_steps
        t_grid = torch.linspace(0.0, 1.0, timesteps + 2, device=self.cfg.device)[:-1]
        t_grid = t_grid**self.cfg.sampler.rho
        if self.cfg.sampler.times_mode == "cosine":
            t_grid = 1 - torch.cos(t_grid * torch.pi / 2.0)

        return t_grid

    def sample(self, n, labels=None, initial_noise=None):
        self.eval()
        device = self.device
        if labels is not None:
            labels = labels.to(self.device)

        t_grid = self.sampling_times()
        self.precompute_sample_metadata(t_grid)
        x = (
            initial_noise.to(device)
            if initial_noise is not None
            else torch.randn(n, 2, device=device)
        )  # Noise
        steps = [x]

        with torch.no_grad():
            for i in tqdm(range(self.cfg.sampler.num_steps, 0, -1)):
                vector = self.get_vector_field(x, t_grid[i], labels=labels)
                x = self.step_along_vector(x, i, vector)
                steps.append(x)

                steps.append(x)

        return x, steps

    def step_along_vector(self, x, i, vector):
        raise NotImplementedError

    def precompute_sample_metadata(self, t_grid):
        pass

    def get_vector_field(self, x, t, labels=None):
        t = t.expand(x.shape[0])
        eps_cond = self.model(x, t, labels)
        eps_uncond = self.model(x, t)
        w = self.cfg.sampler.cfg_omega
        eps = w * eps_cond + (1 - w) * eps_uncond
        return eps

    def get_gt_vf(self, noised, t):
        raise NotImplementedError
