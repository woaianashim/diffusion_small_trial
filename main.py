import hydra
import torch
from src.diffusion import Diffusion
from src.visualizer import Visualizer


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg):
    print(cfg)
    diffusion = Diffusion(cfg)
    visualizer = Visualizer()
    if cfg.checkpoint:
        diffusion.load_state_dict(torch.load(cfg.checkpoint))
    if cfg.mode == "train":
        diffusion.run_train()
    if cfg.mode == "eval":
        batch = diffusion.data.sample_batch(10000)
        red_label = torch.zeros((4, 4))
        green_label = torch.zeros((4, 4))
        # red_label[0, 2] = 1
        # green_label[1, 3] = 1
        green_label = diffusion.data.masked_labels[0]
        batch["label"][:5000] = red_label.view(-1)
        batch["label"][5000:] = green_label.view(-1)
        _, steps = diffusion.sample(10000, batch["label"])
        visualizer.save_trajectory(
            torch.stack(steps, dim=0),
            batch["label"],
            label_colors={red_label: "white", green_label: "green"},
        )


if __name__ == "__main__":
    main()
