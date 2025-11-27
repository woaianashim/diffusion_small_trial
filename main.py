import hydra
import torch
from src.visualizer import Visualizer


@hydra.main(config_path="config", config_name="conf", version_base=None)
def main(cfg):
    print(cfg)
    if cfg.algo == "fm":
        from src.fm import FlowMatching

        algo = FlowMatching(cfg)
    elif cfg.algo == "diffusion":
        from src.diffusion import Diffusion

        algo = Diffusion(cfg)
    else:
        raise NotImplementedError(f"Algo should be fm or diffusion")
    visualizer = Visualizer(algo)
    if cfg.mode == "train":
        if cfg.checkpoint is not None:
            algo.load(cfg.checkpoint)
        algo.run_train()
    if cfg.mode == "eval":
        algo.load(cfg.checkpoint)
        batch = algo.data.sample_batch(10000)
        no_label = torch.zeros((4, 4))
        green_label = torch.zeros((4, 4))
        green_label = algo.data.masked_labels[0]
        batch["label"][:5000] = no_label.view(-1)
        batch["label"][5000:] = green_label.view(-1)
        _, steps = algo.sample(10000, batch["label"])
        visualizer.save_trajectory(
            torch.stack(steps, dim=0),
            batch["label"],
            label_colors={no_label: "white", green_label: "green"},
        )


if __name__ == "__main__":
    main()
