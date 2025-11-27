import torch
from torchvision import io, utils
from torchvision.utils import draw_keypoints


class Visualizer:
    def __init__(
        self,
        algo,
        image_size=512,
        point_size=1,
        bg_color=(0, 0, 0),
        point_color=(255, 255, 255),
        video_path="trajectory.mp4",
        animation_fps=25,
        last_frame_path="last_step.png",
    ):
        self.algo = algo
        self.image_size = image_size
        self.point_size = point_size
        self.bg_color = torch.tensor(bg_color, dtype=torch.uint8)
        self.point_color = point_color
        self.video_path = video_path
        self.animation_fps = animation_fps
        self.last_frame_path = last_frame_path

    def trajectories_to_frames(
        self, points: torch.Tensor, labels=None, label_colors=dict()
    ) -> torch.Tensor:
        T, N, _ = points.shape
        scale = (points.max() - points.min()).clamp(min=1e-6)
        points = (points - points.min()) / scale * self.image_size

        frames = torch.ones(
            (T, 3, self.image_size, self.image_size), dtype=torch.uint8
        ) * self.bg_color.view(3, 1, 1)
        # Fill background
        for t in range(T):

            mask = torch.ones((points.shape[1],), dtype=torch.bool)
            if labels is not None:
                for label, color in label_colors.items():
                    label_mask = self._label_mask(labels, label)
                    frames[t] = draw_keypoints(
                        image=frames[t],
                        keypoints=points[t, label_mask][None],
                        colors=color,
                        radius=self.point_size,
                    )
                    mask = mask * (~label_mask)
            frames[t] = draw_keypoints(
                image=frames[t],
                keypoints=points[t, mask][None],
                colors=self.point_color,
                radius=self.point_size,
            )
        return frames

    def save_trajectory(self, points, labels=None, label_colors=dict()):
        frames = self.trajectories_to_frames(points, labels, label_colors)

        video_frames = frames.permute(0, 2, 3, 1).contiguous()
        io.write_video(
            filename=self.video_path,
            video_array=video_frames,
            fps=self.animation_fps,
        )
        last_frame = frames[-1]
        utils.save_image(last_frame.float() / 255.0, self.last_frame_path)

    def _label_mask(self, labels, label):
        labels = labels.view(labels.shape[0], *label.shape)
        zeros = label.sum(-1) == 0
        return ((labels == label).all(-1) | zeros).all(-1)
