import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import DenseNet
from .data import FracTree
import logging


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.data = FracTree(cfg.data)
        self.model = DenseNet(cfg.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.optim)
        self.loss = torch.nn.MSELoss()
        self.to(cfg.device)

    def run_train(self):
        self.train()

        dataloader = DataLoader(self.data, **self.cfg.dataloader)
        logging.info("Starting training.")
        for ep in range(self.cfg.epochs):
            seen = 0
            epoch_loss = 0.0
            for batch in tqdm(dataloader):
                seen += len(batch["t"])
                noised, t, label, noise = (
                    batch["noised"].to(self.device),
                    batch["t"].to(self.device),
                    batch["label"].to(self.device),
                    batch["noise"].to(self.device),
                )
                label_mask = torch.rand_like(label) > 0.5
                preds = self.model(noised, t, label * label_mask)
                loss = self.loss(preds, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.detach().cpu().item()
            logging.info(f"Average Epoch {ep} loss: {epoch_loss/(seen+1)}")
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
                eps_theta = self.model(x, t, labels)

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
