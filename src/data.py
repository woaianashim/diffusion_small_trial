import math
import torch
from torch.utils.data import IterableDataset
from torch.distributions import Categorical, Beta, Normal
from typing import List, Tuple

__all__ = ["FracTree"]


class FracTree(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.generate_tree()

        lengthes = (self.edges[:, 0] - self.edges[:, 1]).norm(dim=-1)
        self.edge_dist = Categorical(probs=lengthes / lengthes.sum())
        self.beta = Beta(cfg.sampler.beta_alpha, cfg.sampler.beta_beta)
        self.normal = Normal(0, 1)
        self._pre_sample_point()

    def __iter__(self):
        for _ in range(self.cfg.sampler.batches_per_epoch):
            yield self.sample_batch(self.cfg.sampler.batch_size)

    def sample_batch(self, n):
        points, labels = self.sample_points(n)
        t = torch.rand((n,))
        s = self.cfg.sampler.s
        alpha_bar = (torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2)[:, None]
        noise = Normal(0, 1).sample((n, 2))
        points_noised = points * alpha_bar.sqrt() + noise * (1 - alpha_bar).sqrt()
        batch = {
            "gt": points,
            "noised": points_noised,
            "noise": noise,
            "t": t,
            "label": labels,
        }
        return batch

    def _pre_sample_point(self):
        n_pre_sampled = int(self.cfg.sampler.pre_sample)
        if n_pre_sampled > 0:
            indeces = self.edge_dist.sample((n_pre_sampled,))
            s = self.beta.sample((n_pre_sampled,))[..., None]
            self.pre_sampled_points = self.edges[indeces, 0] * s + self.edges[
                indeces, 1
            ] * (1 - s)
            self.pre_sampled_labels = self.labels[indeces].view((n_pre_sampled, -1))

    def sample_points(self, n):
        n_pre_sampled = int(self.cfg.sampler.pre_sample)
        if n_pre_sampled > 0:
            indeces = torch.randint(low=0, high=n_pre_sampled, size=(n,))
            points = self.pre_sampled_points[indeces]
            labels = self.pre_sampled_labels[indeces]

            return points, labels

        else:
            indeces = self.edge_dist.sample((n,))
            s = self.beta.sample((n,))[..., None]
            point = self.edges[indeces, 0] * s + self.edges[indeces, 1] * (1 - s)
            return point, self.labels[indeces].view(n, -1)

    def generate_tree(self):
        self.radii: List[float] = [0.0]
        nodes = []
        edges = []
        label_masks = []
        labels = []
        base_label = torch.zeros((self.cfg.tree.depth, self.cfg.tree.branching + 1))
        base_label[:, 0] = 1
        acc = 0.0

        for lvl in range(self.cfg.tree.depth):
            acc += self.cfg.tree.L0 * (self.cfg.tree.k**lvl)
            self.radii.append(acc)
        root_node = (0, 0.0)
        label_mask = (
            torch.tensor(self.cfg.tree.mask)
            if self.cfg.tree.mask
            else torch.zeros((self.cfg.tree.depth))
        )

        def rec(
            level: int,
            theta_left: float,
            theta_right: float,
            parent_xy: Tuple[float, float],
            label: torch.Tensor,
            mask: torch.Tensor,
        ):
            theta_c = 0.5 * (theta_left + theta_right)
            r = self.radii[level]
            x = root_node[0] + r * math.cos(math.pi * theta_c)
            y = root_node[1] + r * math.sin(math.pi * theta_c)
            nodes.append((x, y))

            if level > 0:
                labels.append(label)
                edges.append((parent_xy, (x, y)))
                masked = (mask == 0).all()
                label_masks.append(masked)

            if level == self.cfg.tree.depth:
                return

            # split sector among children with small gaps
            sector = (theta_right - theta_left) * self.cfg.tree.sector_decay
            base_left = theta_c - sector / 2.0
            sub = sector / self.cfg.tree.branching
            gap = sub * self.cfg.tree.gap_ratio

            for i in range(self.cfg.tree.branching):
                child_left = base_left + i * sub + gap
                child_right = base_left + (i + 1) * sub - gap
                child_label = label.clone()
                child_label[level, 0] = 0
                child_label[level, i + 1] = 1
                child_mask = mask.clone()
                child_mask[level] = i + 1 != child_mask[level]
                rec(level + 1, child_left, child_right, (x, y), child_label, child_mask)

        # start recursion; root has no parent edge
        rec(
            0,
            self.cfg.tree.root_angle - self.cfg.tree.fan / 2.0,
            self.cfg.tree.root_angle + self.cfg.tree.fan / 2.0,
            root_node,
            base_label,
            label_mask,
        )
        self.label_masks = torch.tensor(label_masks)
        self.all_edges = torch.tensor(edges)
        self.all_labels = torch.stack(labels, 0)
        self.edges = self.all_edges[~self.label_masks]
        self.labels = self.all_labels[~self.label_masks]
        self.masked_labels = self.all_labels[self.label_masks]
        self.nodes = torch.tensor(nodes)
