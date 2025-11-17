import hydra
import torch
from src.diffusion import Diffusion


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg):
    print(cfg)
    diffusion = Diffusion(cfg)
    if cfg.checkpoint:
        diffusion.load_state_dict(torch.load(cfg.checkpoint))
    if cfg.mode == "train":
        diffusion.run_train()
    if cfg.mode == "eval":
        batch = diffusion.data.sample_batch(1000)
        _, steps = diffusion.sample(1000, batch["label"])
        diffusion.save_trajectory(torch.stack(steps, dim=0))


if __name__ == "__main__":
    main()
