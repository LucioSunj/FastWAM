import hashlib
import os
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc
from accelerate import PartialState

logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _p1_camera_float_to_uint8(
    video: torch.Tensor,
    *,
    range_tolerance: float,
) -> torch.Tensor:
    """Convert processed RGB floats after rejecting material range violations."""

    if not torch.is_floating_point(video):
        raise TypeError("Processed P1 camera source must be floating point.")
    tolerance = float(range_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("P1 camera range tolerance must be finite and non-negative.")
    if not bool(torch.isfinite(video).all().item()):
        raise ValueError("Processed P1 camera source must contain only finite values.")
    minimum = float(video.amin().item())
    maximum = float(video.amax().item())
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            "Processed P1 camera source exceeds the allowed [0,1] range "
            f"tolerance: min={minimum}, max={maximum}, tolerance={tolerance}."
        )
    return video.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)


class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",  # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[
            str
        ] = None,  # whether to hardcode a specific instruction for all samples, for debugging
        return_p1_camera_uint8: bool = False,
        p1_camera_ids: Optional[list[str]] = None,
        p1_camera_range_tolerance: float = 0.0,
        return_visual_camera_uint8: bool = False,
        visual_camera_ids: Optional[list[str]] = None,
        visual_camera_input_size: int | None = None,
        image_current_frame_only: bool = False,
    ):
        self.image_current_frame_only = bool(image_current_frame_only)
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_current_frame_only=self.image_current_frame_only,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        )
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, (
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        )
        self.video_sample_indices = list(
            range(0, num_frames, self.action_video_freq_ratio)
        )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.return_p1_camera_uint8 = bool(return_p1_camera_uint8)
        self.p1_camera_ids = tuple(str(value) for value in (p1_camera_ids or ()))
        self.p1_camera_range_tolerance = float(p1_camera_range_tolerance)
        self.return_visual_camera_uint8 = bool(return_visual_camera_uint8)
        self.visual_camera_ids = tuple(
            str(value) for value in (visual_camera_ids or ())
        )
        self.visual_camera_input_size = (
            None if visual_camera_input_size is None else int(visual_camera_input_size)
        )
        if self.return_p1_camera_uint8:
            if not self.p1_camera_ids:
                raise ValueError(
                    "P1 camera uint8 output requires an explicit camera order."
                )
            if len(set(self.p1_camera_ids)) != len(self.p1_camera_ids):
                raise ValueError("P1 camera identifiers must be unique and ordered.")
            if (
                not np.isfinite(self.p1_camera_range_tolerance)
                or self.p1_camera_range_tolerance < 0.0
            ):
                raise ValueError(
                    "P1 camera range tolerance must be finite and non-negative."
                )
        if self.return_visual_camera_uint8:
            if self.visual_camera_input_size not in {224, 512}:
                raise ValueError("V2 visual input size must be 224 or 512.")
            if not self.visual_camera_ids or len(set(self.visual_camera_ids)) != len(
                self.visual_camera_ids
            ):
                raise ValueError(
                    "V2 visual camera IDs must be non-empty, unique, and ordered."
                )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )
                else:
                    dataset_stats = None
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(
                        dataset_stats, os.path.join(work_dir, "dataset_stats.json")
                    )

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        image_is_pad = sample["image_is_pad"]
        visual_camera_payload = None
        if self.return_visual_camera_uint8:
            required_visual = {
                "visual_camera_pixels",
                "visual_camera_valid_mask",
                "visual_camera_source_resolution",
                "visual_camera_ids",
            }
            missing_visual = sorted(required_visual - set(sample))
            if missing_visual:
                raise KeyError(
                    "V2 processor did not return raw visual camera fields: "
                    f"{missing_visual}."
                )
            if tuple(sample["visual_camera_ids"]) != self.visual_camera_ids:
                raise ValueError("V2 processor/dataset camera order differs.")
            pixels = sample["visual_camera_pixels"]
            if (
                pixels.shape
                != (
                    len(self.visual_camera_ids),
                    3,
                    self.visual_camera_input_size,
                    self.visual_camera_input_size,
                )
                or pixels.dtype != torch.uint8
            ):
                raise ValueError("V2 processor camera tensor contract changed.")
            visual_camera_payload = {
                "visual_camera_pixels": pixels,
                "visual_camera_valid_mask": sample["visual_camera_valid_mask"],
                "visual_camera_source_resolution": sample[
                    "visual_camera_source_resolution"
                ],
                "visual_camera_ids": self.visual_camera_ids,
            }

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if self.image_current_frame_only:
                if video.shape[1] != 1:
                    raise ValueError(
                        "Current-frame-only image loading requires exactly one "
                        f"decoded frame, got {video.shape[1]}."
                    )
                video = video.expand(
                    -1,
                    len(self.video_sample_indices),
                    -1,
                    -1,
                    -1,
                )
            else:
                video = video[
                    :, self.video_sample_indices, :, :, :
                ]  # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, (
                f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            )
            if self.image_current_frame_only:
                if video.shape[0] != 1:
                    raise ValueError(
                        "Current-frame-only image loading requires exactly one "
                        f"decoded frame, got {video.shape[0]}."
                    )
                video = video.expand(
                    len(self.video_sample_indices),
                    -1,
                    -1,
                    -1,
                )
            else:
                video = video[self.video_sample_indices, :, :, :]  # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if self.image_current_frame_only:
            if image_is_pad.shape != (1,):
                raise ValueError(
                    "Current-frame-only image padding must contain one entry."
                )
            image_is_pad = image_is_pad.expand(len(self.video_sample_indices))
        else:
            image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(
            num_cameras, T_video, C, H, W
        )  # [num_cameras, T_video, C, H, W]
        p1_camera_pixels = None
        p1_camera_valid_mask = None
        if self.return_p1_camera_uint8:
            if num_cameras != len(self.p1_camera_ids):
                raise ValueError(
                    "P1 camera tensor/order mismatch: "
                    f"tensor_views={num_cameras}, camera_ids={self.p1_camera_ids}."
                )
            p1_camera_pixels = _p1_camera_float_to_uint8(
                video[:, 0],
                range_tolerance=self.p1_camera_range_tolerance,
            )
            source_camera_count = len(self.lerobot_dataset.image_meta)
            p1_camera_valid_mask = torch.arange(num_cameras) < source_camera_count
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat(
                    [video[i] for i in range(num_cameras)], dim=-1
                )  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat(
                    [video[i] for i in range(num_cameras)], dim=-2
                )  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"]  # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :]  # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video.shape)}"
            )
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]

        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.return_p1_camera_uint8:
            data.update(
                p1_camera_pixels=p1_camera_pixels,
                p1_camera_valid_mask=p1_camera_valid_mask,
                p1_camera_ids=self.p1_camera_ids,
            )
        if visual_camera_payload is not None:
            data.update(visual_camera_payload)
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(
                f"Error processing sample idx {idx}: {e}. Returning a random sample instead."
            )
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
