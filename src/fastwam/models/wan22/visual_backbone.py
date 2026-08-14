"""Registered, frozen V2 spatial-patch visual backbones.

This module is deliberately local-only.  Asset acquisition is handled by the
explicit preparation script; runtime construction verifies source and weight
identities before allocating a model and never contacts a network service.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from fastwam.adapters import PolicyRegime

from .visual_contracts import (
    PreparedVisualCameraBatch,
    SpatialPatchMemory,
    VISUAL_PATCH_SIZE,
    contract_sha256,
    spatial_patch_layout_contract,
    validate_sha256,
)

PINNED_DINOV3_SOURCE_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
PINNED_LINGBOT_VISION_SOURCE_REVISION = "151e46321bae4399f8568829f190c7bdec216b49"

VISUAL_BACKBONE_ASSET_SCHEMA = "fastwam-visual-backbone-asset-v2"
VISUAL_BACKBONE_METADATA_SCHEMA = "fastwam-visual-backbone-metadata-v2"
VISUAL_MEMORY_SCHEMA = "fastwam-spatial-patch-memory-v2"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class VisualBackbonePreset:
    """One immutable family/variant registration."""

    family: str
    variant: str
    model_name: str
    native_dim: int
    depth: int
    allowed_input_sizes: tuple[int, ...]
    source_revision: str
    weights_repo_id: str
    weights_revision: str
    weights_filename: str
    weights_sha256: str
    weights_size_bytes: int
    license_id: str
    config_file: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.family, self.variant


_PRESETS = (
    VisualBackbonePreset(
        family="dinov3",
        variant="vits16",
        model_name="dinov3_vits16",
        native_dim=384,
        depth=12,
        allowed_input_sizes=(224,),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        weights_revision="114c1379950215c8b35dfcd4e90a5c251dde0d32",
        weights_filename="model.safetensors",
        weights_sha256=(
            "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
        ),
        weights_size_bytes=86406384,
        license_id="DINOv3 License",
    ),
    VisualBackbonePreset(
        family="dinov3",
        variant="vitb16",
        model_name="dinov3_vitb16",
        native_dim=768,
        depth=12,
        allowed_input_sizes=(224,),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        weights_revision="5931719e67bbdb9737e363e781fb0c67687896bc",
        weights_filename="model.safetensors",
        weights_sha256=(
            "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
        ),
        weights_size_bytes=342662192,
        license_id="DINOv3 License",
    ),
    VisualBackbonePreset(
        family="dinov3",
        variant="vitl16",
        model_name="dinov3_vitl16",
        native_dim=1024,
        depth=24,
        allowed_input_sizes=(224,),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        weights_revision="ea8dc2863c51be0a264bab82070e3e8836b02d51",
        weights_filename="model.safetensors",
        weights_sha256=(
            "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        ),
        weights_size_bytes=1212559808,
        license_id="DINOv3 License",
    ),
    VisualBackbonePreset(
        family="lingbot_vision",
        variant="small",
        model_name="vit_small",
        native_dim=384,
        depth=12,
        allowed_input_sizes=(224, 512),
        source_revision=PINNED_LINGBOT_VISION_SOURCE_REVISION,
        weights_repo_id="robbyant/lingbot-vision-vit-small",
        weights_revision="127cbcec380de0bcd55bdc1b1fad3819850a6514",
        weights_filename="model.pt",
        weights_sha256=(
            "dca36562cb6b0b34504df6edc18fa282c5ef06fb375c3e91d5487247a1096f9d"
        ),
        weights_size_bytes=86523066,
        license_id="Apache-2.0",
        config_file="lingbot_vision/configs/lbot_vision_vits.yaml",
    ),
    VisualBackbonePreset(
        family="lingbot_vision",
        variant="base",
        model_name="vit_base",
        native_dim=768,
        depth=12,
        allowed_input_sizes=(224, 512),
        source_revision=PINNED_LINGBOT_VISION_SOURCE_REVISION,
        weights_repo_id="robbyant/lingbot-vision-vit-base",
        weights_revision="f606f8c6c4002234ea68038f4d7c7cf57da96dfa",
        weights_filename="model.pt",
        weights_sha256=(
            "783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825"
        ),
        weights_size_bytes=342852282,
        license_id="Apache-2.0",
        config_file="lingbot_vision/configs/lbot_vision_vitb.yaml",
    ),
    VisualBackbonePreset(
        family="lingbot_vision",
        variant="large",
        model_name="vit_large",
        native_dim=1024,
        depth=24,
        allowed_input_sizes=(224, 512),
        source_revision=PINNED_LINGBOT_VISION_SOURCE_REVISION,
        weights_repo_id="robbyant/lingbot-vision-vit-large",
        weights_revision="5e0370623d4fa5db945d00bc47a8545eed407d6b",
        weights_filename="model.pt",
        weights_sha256=(
            "5b5eb67ebbf990b747658ecf90f1cf2b93f5b0e8dfdfbceb060ae1fc364deb8f"
        ),
        weights_size_bytes=1213035142,
        license_id="Apache-2.0",
        config_file="lingbot_vision/configs/lbot_vision_vitl.yaml",
    ),
)
VISUAL_BACKBONE_REGISTRY = {preset.key: preset for preset in _PRESETS}


def registered_visual_backbones() -> tuple[VisualBackbonePreset, ...]:
    """Return all supported registrations in stable order."""

    return _PRESETS


def get_visual_backbone_preset(
    family: str,
    variant: str,
    *,
    input_size: int,
) -> VisualBackbonePreset:
    """Resolve an exact registration and reject unsupported variants/sizes."""

    key = (str(family).strip().lower(), str(variant).strip().lower())
    if key not in VISUAL_BACKBONE_REGISTRY:
        allowed = ", ".join(f"{item.family}/{item.variant}" for item in _PRESETS)
        raise ValueError(
            f"Unsupported visual backbone {key[0]}/{key[1]}; expected {allowed}."
        )
    preset = VISUAL_BACKBONE_REGISTRY[key]
    size = int(input_size)
    if size not in preset.allowed_input_sizes:
        raise ValueError(
            f"{preset.family}/{preset.variant} input size {size} is unsupported; "
            f"expected {preset.allowed_input_sizes}."
        )
    return preset


def visual_preprocess_contract(
    preset: VisualBackbonePreset,
    *,
    input_size: int,
) -> dict[str, Any]:
    """Return the frozen ImageNet-normalization contract for one preset."""

    return {
        "schema": "fastwam-visual-backbone-preprocess-v2",
        "family": preset.family,
        "variant": preset.variant,
        "color_space": "RGB",
        "input_dtype": "uint8",
        "input_layout": "B,V,C,H,W",
        "input_size": [int(input_size), int(input_size)],
        "spatial_transform": "none_already_resized_by_caller",
        "float_scale": "uint8_div_255",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def visual_output_contract(
    preset: VisualBackbonePreset,
    *,
    input_size: int,
) -> dict[str, Any]:
    """Return the normalized native-patch output contract."""

    grid = int(input_size) // VISUAL_PATCH_SIZE
    return {
        "schema": "fastwam-visual-backbone-native-output-v2",
        "family": preset.family,
        "variant": preset.variant,
        "model": preset.model_name,
        "api": "forward_features",
        "output_key": "x_norm_patchtokens",
        "token_kind": "normalized_spatial_patches",
        "include_cls": False,
        "include_registers": False,
        "patch_size": VISUAL_PATCH_SIZE,
        "patch_grid": [grid, grid],
        "patch_count": grid * grid,
        "native_dim": preset.native_dim,
    }


@dataclass(frozen=True)
class VisualBackboneAssetSpec:
    """Strict local runtime identity for one registered V2 visual backbone."""

    family: str
    variant: str
    input_size: int
    source_root: Path | str
    source_revision: str
    weights_revision: str
    weights_path: Path | str
    weights_sha256: str
    preprocess_sha256: str
    output_contract_sha256: str
    compute_dtype: str
    encode_microbatch_size: int
    license_id: str

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        variant = str(self.variant).strip().lower()
        input_size = int(self.input_size)
        preset = get_visual_backbone_preset(
            family,
            variant,
            input_size=input_size,
        )
        source_root = Path(self.source_root).expanduser().resolve()
        weights_path = Path(self.weights_path).expanduser().resolve()
        source_revision = str(self.source_revision).strip().lower()
        if source_revision != preset.source_revision:
            raise ValueError("Visual source revision does not match its registration.")
        weights_revision = str(self.weights_revision).strip().lower()
        if weights_revision != preset.weights_revision:
            raise ValueError("Visual weights revision does not match its registration.")
        weights_sha256 = validate_sha256(
            self.weights_sha256,
            label="Visual backbone weights SHA256",
        )
        if weights_sha256 != preset.weights_sha256:
            raise ValueError("Visual weights SHA256 does not match its registration.")
        preprocess_sha256 = validate_sha256(
            self.preprocess_sha256,
            label="Visual preprocess SHA256",
        )
        expected_preprocess = contract_sha256(
            visual_preprocess_contract(preset, input_size=input_size)
        )
        if preprocess_sha256 != expected_preprocess:
            raise ValueError("Visual preprocessing contract hash mismatch.")
        output_sha256 = validate_sha256(
            self.output_contract_sha256,
            label="Visual output contract SHA256",
        )
        expected_output = contract_sha256(
            visual_output_contract(preset, input_size=input_size)
        )
        if output_sha256 != expected_output:
            raise ValueError("Visual native-output contract hash mismatch.")
        compute_dtype = str(self.compute_dtype).strip().lower()
        if compute_dtype not in _DTYPES:
            raise ValueError(
                f"Unsupported visual compute dtype {compute_dtype!r}; "
                f"expected {sorted(_DTYPES)}."
            )
        encode_microbatch_size = int(self.encode_microbatch_size)
        if encode_microbatch_size < 1:
            raise ValueError("Visual encode microbatch size must be positive.")
        license_id = str(self.license_id).strip()
        if license_id != preset.license_id:
            raise ValueError("Visual license ID does not match its registration.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "input_size", input_size)
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "weights_revision", weights_revision)
        object.__setattr__(self, "weights_path", weights_path)
        object.__setattr__(self, "weights_sha256", weights_sha256)
        object.__setattr__(self, "preprocess_sha256", preprocess_sha256)
        object.__setattr__(self, "output_contract_sha256", output_sha256)
        object.__setattr__(self, "compute_dtype", compute_dtype)
        object.__setattr__(
            self,
            "encode_microbatch_size",
            encode_microbatch_size,
        )
        object.__setattr__(self, "license_id", license_id)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> VisualBackboneAssetSpec:
        """Parse the exact V2 asset mapping without aliases or unknown fields."""

        required = {
            "family",
            "variant",
            "input_size",
            "source_root",
            "source_revision",
            "weights_revision",
            "weights_path",
            "weights_sha256",
            "preprocess_sha256",
            "output_contract_sha256",
            "compute_dtype",
            "encode_microbatch_size",
            "license_id",
        }
        if set(payload) != required:
            raise ValueError(
                "Invalid visual asset fields; "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}."
            )
        return cls(**dict(payload))

    @property
    def preset(self) -> VisualBackbonePreset:
        return get_visual_backbone_preset(
            self.family,
            self.variant,
            input_size=self.input_size,
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.compute_dtype]

    @property
    def grid(self) -> tuple[int, int]:
        size = self.input_size // VISUAL_PATCH_SIZE
        return size, size

    @property
    def patch_count(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def asset_contract_sha256(self) -> str:
        return contract_sha256(
            {
                "schema": VISUAL_BACKBONE_ASSET_SCHEMA,
                "family": self.family,
                "variant": self.variant,
                "model_name": self.preset.model_name,
                "native_dim": self.preset.native_dim,
                "depth": self.preset.depth,
                "input_size": self.input_size,
                "patch_size": VISUAL_PATCH_SIZE,
                "source_revision": self.source_revision,
                "weights_repo_id": self.preset.weights_repo_id,
                "weights_revision": self.weights_revision,
                "weights_sha256": self.weights_sha256,
                "preprocess_sha256": self.preprocess_sha256,
                "output_contract_sha256": self.output_contract_sha256,
                "compute_dtype": self.compute_dtype,
                "encode_microbatch_size": self.encode_microbatch_size,
                "license_id": self.license_id,
            }
        )

    def checkpoint_metadata(
        self,
        *,
        camera_ids: tuple[str, ...],
        input_contract_sha256: str,
    ) -> dict[str, Any]:
        """Return the complete identity stored by V2 project checkpoints."""

        memory_hash = spatial_memory_contract_sha256(
            self,
            camera_ids=camera_ids,
            input_contract_sha256=input_contract_sha256,
        )
        return {
            "schema": VISUAL_BACKBONE_METADATA_SCHEMA,
            "family": self.family,
            "variant": self.variant,
            "model_name": self.preset.model_name,
            "input_size": self.input_size,
            "patch_size": VISUAL_PATCH_SIZE,
            "patch_grid": list(self.grid),
            "patch_count": self.patch_count,
            "native_dim": self.preset.native_dim,
            "depth": self.preset.depth,
            "source_root": str(self.source_root),
            "source_revision": self.source_revision,
            "weights_repo_id": self.preset.weights_repo_id,
            "weights_revision": self.weights_revision,
            "weights_path": str(self.weights_path),
            "weights_sha256": self.weights_sha256,
            "asset_contract_sha256": self.asset_contract_sha256,
            "input_contract_sha256": validate_sha256(
                input_contract_sha256,
                label="Visual input contract SHA256",
            ),
            "preprocess_sha256": self.preprocess_sha256,
            "output_contract_sha256": self.output_contract_sha256,
            "memory_contract_sha256": memory_hash,
            "compute_dtype": self.compute_dtype,
            "encode_microbatch_size": self.encode_microbatch_size,
            "license_id": self.license_id,
        }


def spatial_memory_contract_sha256(
    asset: VisualBackboneAssetSpec,
    *,
    camera_ids: tuple[str, ...],
    input_contract_sha256: str,
) -> str:
    """Bind V2 memory to its exact asset, layout, and caller geometry."""

    input_hash = validate_sha256(
        input_contract_sha256,
        label="Visual input contract SHA256",
    )
    return contract_sha256(
        {
            "schema": VISUAL_MEMORY_SCHEMA,
            "asset_contract_sha256": asset.asset_contract_sha256,
            "family": asset.family,
            "variant": asset.variant,
            "weights_sha256": asset.weights_sha256,
            "input_contract_sha256": input_hash,
            "preprocess_sha256": asset.preprocess_sha256,
            "output_contract_sha256": asset.output_contract_sha256,
            "camera_ids": list(camera_ids),
            "layout": spatial_patch_layout_contract(
                camera_ids,
                grid=asset.grid,
                patch_size=VISUAL_PATCH_SIZE,
            ),
            "crop_orientation_contract_sha256": input_hash,
        }
    )


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_visual_backbone_asset(asset: VisualBackboneAssetSpec) -> None:
    """Verify local source/weight identity without allocating the backbone."""

    if not asset.source_root.is_dir():
        raise FileNotFoundError(asset.source_root)
    if not asset.weights_path.is_file():
        raise FileNotFoundError(asset.weights_path)
    if asset.weights_path.stat().st_size != asset.preset.weights_size_bytes:
        raise ValueError("Visual weights file size does not match its registration.")
    if _file_sha256(asset.weights_path) != asset.weights_sha256:
        raise ValueError("Visual weights file SHA256 mismatch.")
    source_probe = (
        asset.source_root / "hubconf.py"
        if asset.family == "dinov3"
        else asset.source_root / "lingbot_vision" / "__init__.py"
    )
    if not source_probe.is_file():
        raise FileNotFoundError(source_probe)
    try:
        from git import Repo

        actual_revision = Repo(asset.source_root).head.commit.hexsha.lower()
    except Exception as error:
        raise ValueError(
            "Unable to verify the local visual source revision."
        ) from error
    if actual_revision != asset.source_revision:
        raise ValueError(
            f"Visual source revision mismatch: {actual_revision} != "
            f"{asset.source_revision}."
        )
    manifest_path = asset.weights_path.parent.parent / (
        "visual_backbones_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "V2 runtime requires the verified visual-backbone manifest: "
            f"{manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Visual-backbone manifest is unreadable.") from error
    if (
        manifest.get("schema") != "fastwam-visual-backbone-assets-v2"
        or manifest.get("status") != "PASS"
        or manifest.get("runtime_downloads_allowed") is not False
    ):
        raise ValueError("Visual-backbone manifest is incomplete or unsupported.")
    matches = [
        record
        for record in manifest.get("assets", ())
        if isinstance(record, Mapping)
        and record.get("family") == asset.family
        and record.get("variant") == asset.variant
    ]
    if len(matches) != 1:
        raise ValueError("Visual asset is absent or duplicated in the manifest.")
    record = matches[0]
    expected_manifest = {
        "path": str(asset.weights_path),
        "sha256": asset.weights_sha256,
        "size_bytes": asset.preset.weights_size_bytes,
        "repo_id": asset.preset.weights_repo_id,
        "weights_revision": asset.weights_revision,
        "source_commit": asset.source_revision,
        "license": asset.license_id,
    }
    mismatches = {
        key: (value, record.get(key))
        for key, value in expected_manifest.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Visual asset manifest identity mismatch: {mismatches}.")


@contextmanager
def _local_package_import(source_root: Path, module_name: str):
    """Import a pinned local package and reject an already-imported other copy."""

    value = str(source_root)
    sys.path.insert(0, value)
    try:
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root):
            raise ImportError(
                f"Imported {module_name} from {module_path}, not {source_root}."
            )
        yield module
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)


def _load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                "Visual safetensors loading requires safetensors."
            ) from error
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("Visual weights must contain a tensor mapping.")
    return dict(state)


def _convert_hf_dinov3_state(
    state: Mapping[str, torch.Tensor],
    *,
    official_state: Mapping[str, torch.Tensor],
    depth: int,
) -> dict[str, torch.Tensor]:
    """Convert a verified HF DINOv3 S/B/L serialization to official keys."""

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
        for index in range(int(depth))
        for suffix in expected_layer_suffixes
    )
    if set(state) != expected:
        raise ValueError(
            "Hugging Face DINOv3 tensor schema mismatch: "
            f"missing={sorted(expected - set(state))[:16]}, "
            f"unexpected={sorted(set(state) - expected)[:16]}."
        )
    # Keep references to freshly initialized model-only tensors (for example
    # non-persistent architecture state) rather than cloning another complete
    # ViT-L in host memory. Every checkpoint-owned parameter below is replaced.
    converted = dict(official_state)
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
    for index in range(int(depth)):
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
    missing = sorted(set(official_state) - set(converted))
    unexpected = sorted(set(converted) - set(official_state))
    mismatches = {
        name: (tuple(official_state[name].shape), tuple(converted[name].shape))
        for name in set(official_state) & set(converted)
        if converted[name].shape != official_state[name].shape
    }
    if missing or unexpected or mismatches:
        raise ValueError(
            "Converted DINOv3 state does not match the registered architecture: "
            f"missing={missing[:16]}, unexpected={unexpected[:16]}, "
            f"shape_mismatches={dict(list(mismatches.items())[:16])}."
        )
    return converted


def _strict_lingbot_state(ckpt: Any) -> dict[str, torch.Tensor]:
    """Extract only a complete LingBot backbone state from supported wrappers."""

    state = ckpt
    if isinstance(state, Mapping):
        for key in ("teacher", "model_state", "state_dict", "model", "backbone"):
            value = state.get(key)
            if isinstance(value, Mapping):
                state = value
                break
    if not isinstance(state, Mapping):
        raise TypeError("LingBot checkpoint does not contain a tensor mapping.")
    cleaned = {
        str(name).replace("_orig_mod.", ""): tensor for name, tensor in state.items()
    }
    if cleaned and all(name.startswith("backbone.") for name in cleaned):
        cleaned = {name[len("backbone.") :]: tensor for name, tensor in cleaned.items()}
    if not cleaned or not all(
        isinstance(value, torch.Tensor) for value in cleaned.values()
    ):
        raise TypeError("LingBot backbone state must be a non-empty tensor mapping.")
    return cleaned


class FrozenVisualPatchEncoder(nn.Module, ABC):
    """Frozen local-only encoder that emits V2 normalized spatial patches."""

    def __init__(self, *, model: nn.Module, asset: VisualBackboneAssetSpec) -> None:
        super().__init__()
        self.model = model
        self.asset = asset
        preprocess = visual_preprocess_contract(
            asset.preset,
            input_size=asset.input_size,
        )
        self.register_buffer(
            "_mean",
            torch.tensor(preprocess["mean"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(preprocess["std"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        super().train(False)

    def train(self, mode: bool = True) -> FrozenVisualPatchEncoder:
        del mode
        super().train(False)
        self.model.train(False)
        return self

    @property
    def device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        return self._mean.device if parameter is None else parameter.device

    def memory_contract_sha256(
        self,
        *,
        camera_ids: tuple[str, ...],
        input_contract_sha256: str,
    ) -> str:
        return spatial_memory_contract_sha256(
            self.asset,
            camera_ids=camera_ids,
            input_contract_sha256=input_contract_sha256,
        )

    def prepare_memory(
        self,
        regime: PolicyRegime | str,
        camera_batch: PreparedVisualCameraBatch,
    ) -> SpatialPatchMemory | None:
        selected = PolicyRegime.parse(regime)
        if selected is PolicyRegime.IDM:
            return None
        return self(camera_batch)

    def forward(
        self,
        camera_batch: PreparedVisualCameraBatch,
    ) -> SpatialPatchMemory:
        return self.encode(camera_batch)

    @abstractmethod
    def _forward_patch_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        """Return registered normalized patch tokens for one non-empty chunk."""

    def encode(
        self,
        camera_batch: PreparedVisualCameraBatch,
    ) -> SpatialPatchMemory:
        """Encode valid views in deterministic chunks and scatter fixed slots."""

        if camera_batch.input_size != self.asset.input_size:
            raise ValueError("Prepared visual input size differs from the encoder.")
        batch, views = camera_batch.pixels.shape[:2]
        flat_pixels = camera_batch.pixels.reshape(
            batch * views,
            3,
            self.asset.input_size,
            self.asset.input_size,
        )
        flat_valid = camera_batch.camera_valid_mask.reshape(-1).to(
            device=flat_pixels.device
        )
        selected_pixels = flat_pixels[flat_valid].to(device=self.device)
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(
                0, selected_pixels.shape[0], self.asset.encode_microbatch_size
            ):
                pixels = selected_pixels[
                    start : start + self.asset.encode_microbatch_size
                ]
                normalized = pixels.to(dtype=self.asset.torch_dtype).div_(255.0)
                normalized = (
                    normalized - self._mean.to(dtype=normalized.dtype)
                ) / self._std.to(dtype=normalized.dtype)
                output = self._forward_patch_tokens(normalized)
                expected = (
                    pixels.shape[0],
                    self.asset.patch_count,
                    self.asset.preset.native_dim,
                )
                if not isinstance(output, torch.Tensor) or output.shape != expected:
                    raise ValueError(
                        "Visual native-output shape mismatch: "
                        f"expected {expected}, got {getattr(output, 'shape', None)}."
                    )
                chunks.append(output)
            valid_tokens = torch.cat(chunks, dim=0)
            tokens = torch.zeros(
                batch * views,
                self.asset.patch_count,
                self.asset.preset.native_dim,
                dtype=valid_tokens.dtype,
                device=valid_tokens.device,
            )
            tokens[flat_valid.to(device=tokens.device)] = valid_tokens
            inference_tokens = tokens.reshape(
                batch,
                views,
                self.asset.patch_count,
                self.asset.preset.native_dim,
            ).detach()
        with torch.inference_mode(False):
            tokens = inference_tokens.clone().detach()
        camera_valid_mask = camera_batch.camera_valid_mask.to(device=tokens.device)
        patch_valid_mask = camera_valid_mask.unsqueeze(-1).expand(
            -1,
            -1,
            self.asset.patch_count,
        )
        source_resolution = camera_batch.source_resolution
        if source_resolution is not None:
            source_resolution = source_resolution.to(device=tokens.device)
        return SpatialPatchMemory(
            tokens=tokens,
            patch_valid_mask=patch_valid_mask,
            camera_valid_mask=camera_valid_mask,
            camera_ids=camera_batch.camera_ids,
            grid=self.asset.grid,
            patch_size=VISUAL_PATCH_SIZE,
            backbone_family=self.asset.family,
            backbone_variant=self.asset.variant,
            native_dim=self.asset.preset.native_dim,
            source_revision=self.asset.source_revision,
            weights_sha256=self.asset.weights_sha256,
            asset_contract_sha256=self.asset.asset_contract_sha256,
            input_contract_sha256=camera_batch.input_contract_sha256,
            preprocess_sha256=self.asset.preprocess_sha256,
            output_contract_sha256=self.asset.output_contract_sha256,
            memory_contract_sha256=self.memory_contract_sha256(
                camera_ids=camera_batch.camera_ids,
                input_contract_sha256=camera_batch.input_contract_sha256,
            ),
            source_resolution=source_resolution,
        )


class FrozenDinoV3PatchEncoder(FrozenVisualPatchEncoder):
    """Registered DINOv3 S/B/L encoder using strict official/HF loading."""

    @classmethod
    def from_local_asset(
        cls,
        asset: VisualBackboneAssetSpec,
        *,
        device: torch.device | str,
    ) -> FrozenDinoV3PatchEncoder:
        if asset.family != "dinov3":
            raise ValueError("FrozenDinoV3PatchEncoder requires family=dinov3.")
        verify_visual_backbone_asset(asset)
        with _local_package_import(
            asset.source_root,
            "dinov3.hub.backbones",
        ) as backbones:
            factory = getattr(backbones, asset.preset.model_name, None)
            if factory is None:
                raise AttributeError(
                    f"Pinned DINOv3 source has no {asset.preset.model_name!r}."
                )
            model = factory(pretrained=False)
        state = _load_tensor_state(asset.weights_path)
        if "embeddings.patch_embeddings.weight" in state:
            state = _convert_hf_dinov3_state(
                state,
                official_state=model.state_dict(),
                depth=asset.preset.depth,
            )
        model.load_state_dict(state, strict=True)
        model.to(device=torch.device(device), dtype=asset.torch_dtype)
        return cls(model=model, asset=asset).to(device=torch.device(device))

    def _forward_patch_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(normalized)
        if not isinstance(features, Mapping) or "x_norm_patchtokens" not in features:
            raise ValueError("DINOv3 must return x_norm_patchtokens.")
        return features["x_norm_patchtokens"]


class FrozenLingBotVisionPatchEncoder(FrozenVisualPatchEncoder):
    """Registered LingBot-Vision Small/Base/Large encoder with strict loading."""

    @classmethod
    def from_local_asset(
        cls,
        asset: VisualBackboneAssetSpec,
        *,
        device: torch.device | str,
    ) -> FrozenLingBotVisionPatchEncoder:
        if asset.family != "lingbot_vision":
            raise ValueError(
                "FrozenLingBotVisionPatchEncoder requires family=lingbot_vision."
            )
        verify_visual_backbone_asset(asset)
        assert asset.preset.config_file is not None
        config_path = asset.source_root / asset.preset.config_file
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        with _local_package_import(
            asset.source_root,
            "lingbot_vision.build",
        ) as build_module:
            cfg = OmegaConf.load(config_path)
            if str(cfg.student.arch) != asset.preset.model_name:
                raise ValueError("LingBot config architecture differs from registry.")
            cfg.crops.global_crops_size = asset.input_size
            model, embed_dim = build_module.build_backbone_from_cfg(cfg)
        if int(embed_dim) != asset.preset.native_dim:
            raise ValueError(
                "LingBot constructed embedding width differs from registry."
            )
        state = _strict_lingbot_state(_load_tensor_state(asset.weights_path))
        expected_state = model.state_dict()
        missing = sorted(set(expected_state) - set(state))
        unexpected = sorted(set(state) - set(expected_state))
        shape_mismatches = {
            name: (tuple(expected_state[name].shape), tuple(state[name].shape))
            for name in set(expected_state) & set(state)
            if state[name].shape != expected_state[name].shape
        }
        if missing or unexpected or shape_mismatches:
            raise ValueError(
                "LingBot checkpoint must exactly match the registered backbone: "
                f"missing={missing[:16]}, unexpected={unexpected[:16]}, "
                f"shape_mismatches={dict(list(shape_mismatches.items())[:16])}."
            )
        model.load_state_dict(state, strict=True)
        model.to(device=torch.device(device), dtype=asset.torch_dtype)
        return cls(model=model, asset=asset).to(device=torch.device(device))

    def _forward_patch_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        features = self.model(normalized, is_training=True)
        if not isinstance(features, Mapping) or "x_norm_patchtokens" not in features:
            raise ValueError("LingBot-Vision must return x_norm_patchtokens.")
        return features["x_norm_patchtokens"]


def build_frozen_visual_encoder(
    asset: VisualBackboneAssetSpec,
    *,
    device: torch.device | str,
) -> FrozenVisualPatchEncoder:
    """Construct the registered frozen encoder with no runtime networking."""

    if asset.family == "dinov3":
        return FrozenDinoV3PatchEncoder.from_local_asset(asset, device=device)
    if asset.family == "lingbot_vision":
        return FrozenLingBotVisionPatchEncoder.from_local_asset(asset, device=device)
    raise AssertionError(f"Unreachable registered family: {asset.family}")
