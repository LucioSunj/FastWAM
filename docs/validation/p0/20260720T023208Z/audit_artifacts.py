import json
import os
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer


component_root = Path("/root/autodl-fs/fastwam/models/wan22_components")
action_path = Path(
    "/root/When-will-inference-time-prediction-beneficial-/FastWAM/checkpoints/"
    "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
)


def require_readable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.R_OK):
        raise RuntimeError(f"missing or unreadable: {path}")
    print(f"READABLE size={path.stat().st_size} path={path}")


index_path = component_root / "diffusion_pytorch_model.safetensors.index.json"
require_readable(index_path)
index = json.loads(index_path.read_text())
shard_names = sorted(set(index["weight_map"].values()))
print(f"DIT_INDEX tensors={len(index['weight_map'])} shards={len(shard_names)}")
for shard_name in shard_names:
    shard_path = component_root / shard_name
    require_readable(shard_path)
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        print(f"SAFETENSORS_HEADER tensors={len(handle.keys())} path={shard_path}")

for filename in (
    "Wan2.2_VAE.safetensors",
    "models_t5_umt5-xxl-enc-bf16.safetensors",
):
    path = component_root / filename
    require_readable(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        print(f"SAFETENSORS_HEADER tensors={len(handle.keys())} path={path}")

tokenizer_path = component_root / "google/umt5-xxl"
for filename in (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
):
    require_readable(tokenizer_path / filename)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
print(f"TOKENIZER_LOAD class={type(tokenizer).__name__} vocab_size={tokenizer.vocab_size}")

require_readable(action_path.resolve())
with action_path.open("rb") as handle:
    print(f"ACTION_DIT_MAGIC {handle.read(8).hex()}")

print("ARTIFACT_AUDIT PASS")
