import torch
import torch.nn as nn
from torchvision import io, utils
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
            epoch_loss_cond = 0.0
            epoch_loss_uncond = 0.0
            for batch in tqdm(dataloader):
                seen += len(batch["t"])
                noised, t, label, noise = (
                    batch["noised"].to(self.device),
                    batch["t"].to(self.device),
                    batch["label"].to(self.device),
                    batch["noise"].to(self.device),
                )
                preds_cond = self.model(noised, t, label)
                preds_uncond = self.model(noised, t, label * 0)
                loss_cond = self.loss(preds_cond, noise)
                loss_uncond = self.loss(preds_uncond, noise)
                loss = loss_cond + loss_uncond
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss_cond += loss_cond.detach().cpu().item()
                epoch_loss_uncond += loss_uncond.detach().cpu().item()
            logging.info(
                f"Average Epoch {ep} loss:\nConditional:"
                + f"{epoch_loss_cond/(seen+1)}\nUnconditional: {epoch_loss_uncond/(seen+1)}"
            )
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

    def trajectories_to_frames(
        self,
        positions: torch.Tensor,
        image_size=(256, 256),
        point_radius: int = 2,
        bg_color=(0, 0, 0),
        point_color=(255, 255, 255),
    ) -> torch.Tensor:
        T, N, _ = positions.shape
        H, W = image_size

        min_xy = positions.amin(dim=(0, 1))  # (2,)
        max_xy = positions.amax(dim=(0, 1))  # (2,)
        scale = (max_xy - min_xy).clamp(min=1e-6)

        norm = (positions - min_xy) / scale  # (T, N, 2) in [0, 1]

        xs = (norm[..., 0] * (W - 1)).round().long().clamp(0, W - 1)  # (T, N)
        ys = (norm[..., 1] * (H - 1)).round().long().clamp(0, H - 1)  # (T, N)

        frames = torch.zeros((T, 3, H, W), dtype=torch.uint8)

        bg_color = torch.tensor(bg_color, dtype=torch.uint8)
        point_color = torch.tensor(point_color, dtype=torch.uint8)

        # Fill background
        frames[:] = bg_color.view(3, 1, 1)

        for t in range(T):
            for i in range(N):
                cx = xs[t, i].item()
                cy = ys[t, i].item()

                x_min = max(cx - point_radius, 0)
                x_max = min(cx + point_radius, W - 1)
                y_min = max(cy - point_radius, 0)
                y_max = min(cy + point_radius, H - 1)

                # Paint a small square for each point
                for c in range(3):
                    frames[t, c, y_min : y_max + 1, x_min : x_max + 1] = point_color[c]

        return frames

    def save_trajectory(
        self,
        points,
        video_path: str = "trajectory.mp4",
        last_frame_path: str = "last_step.png",
        image_size=(256, 256),
        fps: int = 25,
    ):
        frames = self.trajectories_to_frames(
            points,
            image_size=image_size,
            point_radius=3,
            bg_color=(0, 0, 0),
            point_color=(255, 255, 255),
        )

        video_frames = frames.permute(0, 2, 3, 1).contiguous()
        io.write_video(
            filename=video_path,
            video_array=video_frames,
            fps=fps,
        )
        last_frame = frames[-1]
        utils.save_image(last_frame.float() / 255.0, last_frame_path)
