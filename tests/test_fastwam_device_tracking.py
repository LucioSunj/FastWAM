import torch
import torch.nn as nn

from fastwam.models.wan22.runtime_placement import RuntimePlacementMixin


class _PlacementAwareModule(RuntimePlacementMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Linear(2, 2, bias=False)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32


def test_parent_module_move_updates_fastwam_runtime_metadata():
    actor = _PlacementAwareModule()
    policy = nn.ModuleDict({"actor": actor})

    policy.to(device="meta", dtype=torch.float64)

    assert actor.device == torch.device("meta")
    assert actor.torch_dtype == torch.float64
