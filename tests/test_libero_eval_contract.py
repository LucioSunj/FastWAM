import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

LIBERO_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "libero"
if str(LIBERO_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_EXPERIMENT_ROOT))

import eval_libero_single as libero_eval


class _RecordingModel:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    proprio_dim = 8

    def __init__(self) -> None:
        self.infer_kwargs = None

    def encode_prompt(self, prompt: str):
        del prompt
        return torch.zeros((1, 1, 1)), torch.ones((1, 1), dtype=torch.bool)

    def infer_action(self, *, num_video_frames: int, **kwargs):
        self.infer_kwargs = {"num_video_frames": num_video_frames, **kwargs}
        return {"action": torch.zeros((kwargs["action_horizon"], 7))}


def test_warmup_passes_configured_video_frame_count() -> None:
    cfg = OmegaConf.create(
        {
            "eval_num_inference_steps": 10,
            "data": {"train": {"num_frames": 33, "action_video_freq_ratio": 4}},
            "EVALUATION": {"rand_device": "cpu"},
        }
    )
    model = _RecordingModel()

    libero_eval._warmup_model(
        model, action_horizon=32, input_h=224, input_w=448, cfg=cfg
    )

    assert model.infer_kwargs is not None
    assert model.infer_kwargs["num_video_frames"] == 9
