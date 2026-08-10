"""Pinned, frozen DINOv3 native patch-memory construction."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fastwam.adapters import PolicyRegime

from .visual_contracts import (
    DINO_V3_INPUT_SIZE,
    DINO_V3_NATIVE_DIM,
    DINO_V3_PATCH_COUNT,
    DINO_V3_PATCH_GRID,
    NativePatchMemory,
    PreparedCameraBatch,
    contract_sha256,
    validate_sha256,
)

PINNED_DINOV3_SOURCE_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
PINNED_DINOV3_MODEL_NAME = "dinov3_vits16"
PINNED_DINOV3_WEIGHTS_FAMILY = "LVD1689M"

DINO_V3_PREPROCESS_CONTRACT = {
    "schema": "fastwam-dinov3-preprocess-v1",
    "color_space": "RGB",
    "input_dtype": "uint8",
    "input_layout": "B,V,C,H,W",
    "input_size": [DINO_V3_INPUT_SIZE, DINO_V3_INPUT_SIZE],
    "spatial_transform": "none_already_cropped_by_caller",
    "float_scale": "uint8_div_255",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
DINO_V3_PREPROCESS_SHA256 = contract_sha256(DINO_V3_PREPROCESS_CONTRACT)

DINO_V3_OUTPUT_CONTRACT = {
    "schema": "fastwam-dinov3-native-output-v1",
    "model": PINNED_DINOV3_MODEL_NAME,
    "api": "forward_features",
    "output_key": "x_norm_patchtokens",
    "token_kind": "normalized_spatial_patches",
    "include_cls": False,
    "include_registers": False,
    "patch_grid": list(DINO_V3_PATCH_GRID),
    "patch_count": DINO_V3_PATCH_COUNT,
    "native_dim": DINO_V3_NATIVE_DIM,
}
DINO_V3_OUTPUT_CONTRACT_SHA256 = contract_sha256(DINO_V3_OUTPUT_CONTRACT)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _parse_compute_dtype(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _DTYPES:
        allowed = ", ".join(sorted(_DTYPES))
        raise ValueError(
            f"Unsupported DINOv3 compute dtype {value!r}; expected {allowed}."
        )
    return normalized


@dataclass(frozen=True)
class DinoV3AssetSpec:
    """Strict local-asset contract for the pinned DINOv3 ViT-S/16 model."""

    source_root: Path | str
    source_revision: str
    model_name: str
    weights_path: Path | str
    weights_sha256: str
    preprocess_sha256: str
    output_contract_sha256: str
    compute_dtype: str
    license_id: str

    def __post_init__(self) -> None:
        source_root = Path(self.source_root).expanduser().resolve()
        weights_path = Path(self.weights_path).expanduser().resolve()
        source_revision = str(self.source_revision).strip().lower()
        model_name = str(self.model_name).strip()
        if source_revision != PINNED_DINOV3_SOURCE_REVISION:
            raise ValueError(
                "DINOv3 source revision must match the pinned FastWAM gitlink."
            )
        if model_name != PINNED_DINOV3_MODEL_NAME:
            raise ValueError("P1 requires the pinned dinov3_vits16 architecture.")
        preprocess_sha256 = validate_sha256(
            self.preprocess_sha256,
            label="DINOv3 preprocess SHA256",
        )
        if preprocess_sha256 != DINO_V3_PREPROCESS_SHA256:
            raise ValueError("DINOv3 preprocessing contract hash mismatch.")
        output_contract_sha256 = validate_sha256(
            self.output_contract_sha256,
            label="DINOv3 output contract SHA256",
        )
        if output_contract_sha256 != DINO_V3_OUTPUT_CONTRACT_SHA256:
            raise ValueError("DINOv3 native-output contract hash mismatch.")
        license_id = str(self.license_id).strip()
        if not license_id:
            raise ValueError("DINOv3 license provenance must be recorded.")
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "weights_path", weights_path)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(
            self,
            "weights_sha256",
            validate_sha256(self.weights_sha256, label="DINOv3 weights SHA256"),
        )
        object.__setattr__(self, "preprocess_sha256", preprocess_sha256)
        object.__setattr__(self, "output_contract_sha256", output_contract_sha256)
        object.__setattr__(
            self,
            "compute_dtype",
            _parse_compute_dtype(self.compute_dtype),
        )
        object.__setattr__(self, "license_id", license_id)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DinoV3AssetSpec:
        """Parse a mapping without accepting missing or unknown fields."""

        required = {
            "source_root",
            "source_revision",
            "model_name",
            "weights_path",
            "weights_sha256",
            "preprocess_sha256",
            "output_contract_sha256",
            "compute_dtype",
            "license_id",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            unknown = sorted(set(payload) - required)
            raise ValueError(
                f"Invalid DINOv3 asset fields; missing={missing}, unknown={unknown}."
            )
        return cls(**dict(payload))

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the configured torch compute dtype."""

        return _DTYPES[self.compute_dtype]


def native_memory_contract_sha256(
    asset: DinoV3AssetSpec,
    *,
    camera_ids: tuple[str, ...],
    input_contract_sha256: str,
) -> str:
    """Bind native memory to asset, camera order, and caller geometry."""

    input_hash = validate_sha256(
        input_contract_sha256,
        label="Camera input contract SHA256",
    )
    return contract_sha256(
        {
            "schema": "fastwam-dinov3-native-memory-v1",
            "source_revision": asset.source_revision,
            "model_name": asset.model_name,
            "weights_family": PINNED_DINOV3_WEIGHTS_FAMILY,
            "weights_sha256": asset.weights_sha256,
            "input_contract_sha256": input_hash,
            "preprocess_sha256": asset.preprocess_sha256,
            "output_contract_sha256": asset.output_contract_sha256,
            "camera_ids": list(camera_ids),
            "patch_grid": list(DINO_V3_PATCH_GRID),
        }
    )


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_local_asset(asset: DinoV3AssetSpec) -> None:
    if not asset.source_root.is_dir():
        raise FileNotFoundError(asset.source_root)
    if not (asset.source_root / "hubconf.py").is_file():
        raise FileNotFoundError(asset.source_root / "hubconf.py")
    if not asset.weights_path.is_file():
        raise FileNotFoundError(asset.weights_path)
    if _file_sha256(asset.weights_path) != asset.weights_sha256:
        raise ValueError("DINOv3 weights file SHA256 mismatch.")
    try:
        from git import Repo

        actual_revision = Repo(asset.source_root).head.commit.hexsha.lower()
    except Exception as error:
        raise ValueError("Unable to verify the local DINOv3 git revision.") from error
    if actual_revision != asset.source_revision:
        raise ValueError(
            f"DINOv3 source revision mismatch: {actual_revision} != "
            f"{asset.source_revision}."
        )


@contextmanager
def _local_dinov3_import(source_root: Path):
    """Import only the pinned backbone module from the verified source tree."""

    source_value = str(source_root)
    sys.path.insert(0, source_value)
    try:
        module = importlib.import_module("dinov3.hub.backbones")
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root):
            raise ImportError(f"Imported DINOv3 from {module_path}, not {source_root}.")
        yield module
    finally:
        if sys.path and sys.path[0] == source_value:
            sys.path.pop(0)


def _load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a verified local PyTorch or safetensors flat tensor state."""

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                "Loading DINOv3 safetensors requires the safetensors package."
            ) from error
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise TypeError("DINOv3 weights must be a flat tensor state dictionary.")
    return dict(state)


def _convert_hf_vits16_state(
    state: Mapping[str, torch.Tensor],
    *,
    official_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the verified HF ViT-S/16 serialization to pinned source keys."""

    expected_global = {
        "embeddings.cls_token",
        "embeddings.mask_token",
        "embeddings.patch_embeddings.bias",
        "embeddings.patch_embeddings.weight",
        "embeddings.register_tokens",
        "norm.bias",
        "norm.weight",
    }
    expected_layer_suffixes = {
        "attention.k_proj.weight",
        "attention.o_proj.bias",
        "attention.o_proj.weight",
        "attention.q_proj.bias",
        "attention.q_proj.weight",
        "attention.v_proj.bias",
        "attention.v_proj.weight",
        "layer_scale1.lambda1",
        "layer_scale2.lambda1",
        "mlp.down_proj.bias",
        "mlp.down_proj.weight",
        "mlp.up_proj.bias",
        "mlp.up_proj.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    }
    expected = set(expected_global)
    expected.update(
        f"layer.{index}.{suffix}"
        for index in range(12)
        for suffix in expected_layer_suffixes
    )
    if set(state) != expected:
        raise ValueError(
            "Hugging Face DINOv3 ViT-S/16 tensor schema mismatch: "
            f"missing={sorted(expected - set(state))[:16]}, "
            f"unexpected={sorted(set(state) - expected)[:16]}."
        )

    converted = {
        name: tensor.detach().cpu().clone() for name, tensor in official_state.items()
    }
    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for source, target in direct.items():
        converted[target] = state[source]
    converted["mask_token"] = state["embeddings.mask_token"].squeeze(0)

    for index in range(12):
        source = f"layer.{index}"
        target = f"blocks.{index}"
        for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            converted[f"{target}.{suffix}"] = state[f"{source}.{suffix}"]
        converted[f"{target}.attn.qkv.weight"] = torch.cat(
            (
                state[f"{source}.attention.q_proj.weight"],
                state[f"{source}.attention.k_proj.weight"],
                state[f"{source}.attention.v_proj.weight"],
            ),
            dim=0,
        )
        converted[f"{target}.attn.qkv.bias"] = torch.cat(
            (
                state[f"{source}.attention.q_proj.bias"],
                torch.zeros_like(state[f"{source}.attention.q_proj.bias"]),
                state[f"{source}.attention.v_proj.bias"],
            ),
            dim=0,
        )
        layer_direct = {
            "attention.o_proj.weight": "attn.proj.weight",
            "attention.o_proj.bias": "attn.proj.bias",
            "layer_scale1.lambda1": "ls1.gamma",
            "layer_scale2.lambda1": "ls2.gamma",
            "mlp.up_proj.weight": "mlp.fc1.weight",
            "mlp.up_proj.bias": "mlp.fc1.bias",
            "mlp.down_proj.weight": "mlp.fc2.weight",
            "mlp.down_proj.bias": "mlp.fc2.bias",
        }
        for source_suffix, target_suffix in layer_direct.items():
            converted[f"{target}.{target_suffix}"] = state[f"{source}.{source_suffix}"]

    official_names = set(official_state)
    converted_names = set(converted)
    mismatches = {
        name: (tuple(official_state[name].shape), tuple(converted[name].shape))
        for name in official_names & converted_names
        if converted[name].shape != official_state[name].shape
    }
    if converted_names != official_names or mismatches:
        raise ValueError(
            "Converted DINOv3 state does not match the pinned architecture: "
            f"missing={sorted(official_names - converted_names)[:16]}, "
            f"unexpected={sorted(converted_names - official_names)[:16]}, "
            f"shape_mismatches={dict(list(mismatches.items())[:16])}."
        )
    return converted


class FrozenDinoV3Encoder(nn.Module):
    """Frozen DINOv3 encoder that emits only native normalized patch tokens."""

    def __init__(self, *, model: nn.Module, asset: DinoV3AssetSpec) -> None:
        super().__init__()
        self.model = model
        self.asset = asset
        self.register_buffer(
            "_mean",
            torch.tensor(DINO_V3_PREPROCESS_CONTRACT["mean"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(DINO_V3_PREPROCESS_CONTRACT["std"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        super().train(False)

    @classmethod
    def from_local_asset(
        cls,
        asset: DinoV3AssetSpec,
        *,
        device: torch.device | str,
    ) -> FrozenDinoV3Encoder:
        """Verify and load the pinned source and local weights without networking."""

        _verify_local_asset(asset)
        with _local_dinov3_import(asset.source_root) as backbones:
            factory = getattr(backbones, asset.model_name, None)
            if factory is None:
                raise AttributeError(
                    f"Pinned DINOv3 source has no {asset.model_name!r} factory."
                )
            model = factory(pretrained=False)
        state = _load_tensor_state(asset.weights_path)
        if "embeddings.patch_embeddings.weight" in state:
            state = _convert_hf_vits16_state(
                state,
                official_state=model.state_dict(),
            )
        model.load_state_dict(state, strict=True)
        model.to(device=torch.device(device), dtype=asset.torch_dtype)
        return cls(model=model, asset=asset).to(device=torch.device(device))

    @classmethod
    def _from_preloaded_model_for_tests(
        cls,
        *,
        model: nn.Module,
        asset: DinoV3AssetSpec,
        device: torch.device | str = "cpu",
    ) -> FrozenDinoV3Encoder:
        model.to(device=torch.device(device), dtype=asset.torch_dtype)
        return cls(model=model, asset=asset).to(device=torch.device(device))

    def train(self, mode: bool = True) -> FrozenDinoV3Encoder:
        """Keep the frozen encoder in evaluation mode permanently."""

        del mode
        super().train(False)
        self.model.train(False)
        return self

    @property
    def device(self) -> torch.device:
        """Return the encoder device."""

        parameter = next(self.model.parameters(), None)
        return self._mean.device if parameter is None else parameter.device

    def prepare_memory(
        self,
        regime: PolicyRegime | str,
        camera_batch: PreparedCameraBatch,
    ) -> NativePatchMemory | None:
        """Return no graph for IDM and one reusable native memory for UNCOND."""

        selected = PolicyRegime.parse(regime)
        if selected is PolicyRegime.IDM:
            return None
        return self.encode(camera_batch)

    def encode(self, camera_batch: PreparedCameraBatch) -> NativePatchMemory:
        """Encode valid views independently and scatter them into fixed slots."""

        batch, views = camera_batch.pixels.shape[:2]
        flat_pixels = camera_batch.pixels.reshape(
            batch * views,
            3,
            DINO_V3_INPUT_SIZE,
            DINO_V3_INPUT_SIZE,
        )
        flat_valid = camera_batch.camera_valid_mask.reshape(-1).to(
            device=flat_pixels.device
        )
        selected_pixels = flat_pixels[flat_valid].to(device=self.device)
        with torch.inference_mode():
            normalized = selected_pixels.to(dtype=self.asset.torch_dtype).div_(255.0)
            normalized = (
                normalized - self._mean.to(dtype=normalized.dtype)
            ) / self._std.to(dtype=normalized.dtype)
            features = self.model.forward_features(normalized)
            if (
                not isinstance(features, Mapping)
                or "x_norm_patchtokens" not in features
            ):
                raise ValueError(
                    "Pinned DINOv3 forward_features must return x_norm_patchtokens."
                )
            valid_tokens = features["x_norm_patchtokens"]
            expected = (
                selected_pixels.shape[0],
                DINO_V3_PATCH_COUNT,
                DINO_V3_NATIVE_DIM,
            )
            if (
                not isinstance(valid_tokens, torch.Tensor)
                or valid_tokens.shape != expected
            ):
                raise ValueError(
                    "Pinned DINOv3 native output shape mismatch: "
                    f"expected {expected}, got {getattr(valid_tokens, 'shape', None)}."
                )
            tokens = torch.zeros(
                batch * views,
                DINO_V3_PATCH_COUNT,
                DINO_V3_NATIVE_DIM,
                dtype=valid_tokens.dtype,
                device=valid_tokens.device,
            )
            tokens[flat_valid.to(device=valid_tokens.device)] = valid_tokens
            inference_tokens = tokens.reshape(
                batch,
                views,
                DINO_V3_PATCH_COUNT,
                DINO_V3_NATIVE_DIM,
            ).detach()
        # Downstream trainable readers save native memory for backward.  A
        # detached inference tensor remains illegal there, so materialize
        # ordinary tensor semantics after leaving the frozen encoder scope.
        with torch.inference_mode(False):
            tokens = inference_tokens.clone().detach()
        camera_valid_mask = camera_batch.camera_valid_mask.to(device=tokens.device)
        patch_valid_mask = camera_valid_mask.unsqueeze(-1).expand(
            -1,
            -1,
            DINO_V3_PATCH_COUNT,
        )
        memory_hash = native_memory_contract_sha256(
            self.asset,
            camera_ids=camera_batch.camera_ids,
            input_contract_sha256=camera_batch.input_contract_sha256,
        )
        return NativePatchMemory(
            tokens=tokens,
            patch_valid_mask=patch_valid_mask,
            camera_valid_mask=camera_valid_mask,
            camera_ids=camera_batch.camera_ids,
            grid=DINO_V3_PATCH_GRID,
            source_revision=self.asset.source_revision,
            weights_sha256=self.asset.weights_sha256,
            input_contract_sha256=camera_batch.input_contract_sha256,
            preprocess_sha256=self.asset.preprocess_sha256,
            output_contract_sha256=self.asset.output_contract_sha256,
            memory_contract_sha256=memory_hash,
        )
