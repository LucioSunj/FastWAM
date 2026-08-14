"""Fail-fast single-GPU runner for the P1 DINO BC T1 feasibility smoke."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from fastwam.adapters import RegimeLoRAConfig, sha256_file
from fastwam.models.wan22.dinov3_memory import (
    DinoV3AssetSpec,
    FrozenDinoV3Encoder,
    native_memory_contract_sha256,
)
from fastwam.models.wan22.visual_backbone import (
    FrozenVisualPatchEncoder,
    VisualBackboneAssetSpec,
    build_frozen_visual_encoder,
    verify_visual_backbone_asset,
)
from fastwam.models.wan22.adaptive_action import CachedActionCondition
from fastwam.models.wan22.visual_contracts import (
    NativePatchMemory,
    SpatialPatchMemory,
    contract_sha256,
)
from fastwam.models.wan22.visual_sidecar import (
    DinoSemanticReaderConfig,
    ProjectionSpec,
    VisualPatchReaderConfig,
    build_dino_semantic_reader,
    build_visual_patch_reader,
)
from fastwam.p1_dino_bc import (
    FastWAMP1DinoBCConfig,
    FastWAMP1DinoBCPolicy,
    FastWAMP1VisualBCPolicy,
    build_p1_optimizer,
)
from fastwam.p1_dino_bc_checkpoint import (
    inspect_p1_dino_bc_checkpoint,
    save_p1_dino_bc_checkpoint,
)
from fastwam.uncond_bc import (
    FastWAMUncondBCConfig,
    stateless_validation_flow_inputs,
)
from fastwam.uncond_bc_trainer import load_strict_fastwam_parent
from fastwam.utils import misc

P1_OUTPUT_MARKER = ".fastwam-p1-dino-bc-output-v1"
_FIXED_STAGES = {"t1_smoke", "tiny_overfit", "short_pilot"}
_FIXED_ARMS = {"a0_bc", "a3_joint"}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args],
            text=True,
        ).strip()

    try:
        return {
            "head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current") or "DETACHED",
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": repr(error)}


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def claim_p1_output(cfg: DictConfig) -> Path:
    """Claim a new P1 artifact directory without overwriting prior evidence."""

    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    marker = output_dir / P1_OUTPUT_MARKER
    if output_dir.exists():
        entries = {entry.name for entry in output_dir.iterdir()}
        if entries and P1_OUTPUT_MARKER not in entries:
            raise FileExistsError(f"P1 output directory is not owned: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    if not marker.is_file():
        _atomic_text(marker, "fastwam-p1-dino-bc-output-v1\n")
    _atomic_text(
        output_dir / "resolved_config.yaml", OmegaConf.to_yaml(cfg, resolve=True)
    )
    return output_dir


def validate_p1_config(cfg: DictConfig) -> None:
    stage = str(cfg.runner.stage)
    arm = str(cfg.runner.arm)
    if stage not in _FIXED_STAGES:
        raise ValueError(
            f"Unknown P1 stage {stage!r}; expected {sorted(_FIXED_STAGES)}."
        )
    if arm not in _FIXED_ARMS:
        raise ValueError(f"Unknown P1 arm {arm!r}; expected {sorted(_FIXED_ARMS)}.")
    if stage == "t1_smoke" and arm != "a3_joint":
        raise ValueError("P1 T1 smoke must use the A3-JOINT arm.")
    if int(cfg.seed) != 42 or bool(cfg.get("compile", False)):
        raise ValueError("P1 first feasibility uses seed 42 and eager execution.")
    reader = cfg.p1.reader
    fixed = {
        "camera_ids": ["main", "wrist"],
        "layer_indices": [6, 12, 18, 22],
        "temperature": 0.07,
        "residual_scale": 1.0,
        "position_mode": "native_contextual_only",
    }
    observed = {
        "camera_ids": list(cfg.p1.camera_ids),
        "layer_indices": list(reader.layer_indices),
        "temperature": float(reader.temperature),
        "residual_scale": float(reader.residual_scale),
        "position_mode": str(cfg.p1.position_mode),
    }
    if observed != fixed:
        raise ValueError(f"P1 fixed first configuration changed: {observed}.")
    for name in ("query_projection", "output_projection"):
        projection = reader[name]
        if str(projection.kind) != "full_linear" or projection.rank is not None:
            raise ValueError(f"P1 first {name} must be full_linear with null rank.")
    thresholds = cfg.p1.thresholds
    if (
        float(thresholds.tiny_final_over_initial_max) != 0.20
        or float(thresholds.memory_relative_improvement_min) != 0.10
        or float(thresholds.delta_position_relative_loss) != 0.05
        or int(thresholds.spatial_positive_pairs_min) != 3
        or int(thresholds.spatial_pair_count) != 4
        or float(thresholds.short_a3_over_a0_dev_max) != 1.02
        or float(thresholds.short_correct_over_shuffled_improvement_min) != 0.01
    ):
        raise ValueError("P1 feasibility thresholds changed after preregistration.")
    camera_contract = OmegaConf.to_container(
        cfg.p1.camera_input_contract,
        resolve=True,
    )
    if not isinstance(camera_contract, dict):
        raise TypeError("P1 camera input contract must resolve to a mapping.")
    if contract_sha256(camera_contract) != str(cfg.p1.camera_input_contract_sha256):
        raise ValueError("P1 camera crop/orientation contract hash mismatch.")
    spatial_pairs = OmegaConf.to_container(cfg.p1.spatial_pairs, resolve=True)
    if not isinstance(spatial_pairs, list) or len(spatial_pairs) != 4:
        raise ValueError("P1 requires exactly four preregistered spatial pairs.")
    identities = []
    for pair in spatial_pairs:
        if not isinstance(pair, dict) or set(pair) != {
            "pair_id",
            "suite",
            "task_index",
            "object",
            "left",
            "right",
            "relation",
        }:
            raise ValueError("P1 spatial-pair schema changed.")
        identities.extend(
            (
                json.dumps(pair["left"], sort_keys=True),
                json.dumps(pair["right"], sort_keys=True),
            )
        )
    if len(set(identities)) != 8:
        raise ValueError("P1 spatial-pair endpoints must be eight unique windows.")
    if int(cfg.training.tiny_steps) not in range(200, 501):
        raise ValueError("P1 tiny overfit must run 200 to 500 optimizer steps.")
    if not 8 <= len(cfg.data.tiny_window_indices) <= 16:
        raise ValueError("P1 tiny overfit must freeze 8 to 16 windows.")
    if (
        list(cfg.data.short_train_episode_ids) != [0, 1, 2, 3, 50, 51, 52, 53]
        or list(cfg.data.short_dev_episode_ids) != [4, 5, 54, 55]
        or int(cfg.data.short_frame_index) != 0
    ):
        raise ValueError("P1 short matched episode/frame split changed.")
    if set(cfg.data.short_train_episode_ids) & set(cfg.data.short_dev_episode_ids):
        raise ValueError("P1 short train/dev episodes must be disjoint.")
    if str(cfg.data.dataset_revision) != "libero_mujoco3.3.2+lerobot-v2.1":
        raise ValueError("P1 dataset revision changed after preregistration.")


def _asset_spec(cfg: DictConfig) -> DinoV3AssetSpec:
    payload = OmegaConf.to_container(cfg.p1.dinov3, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("P1 DINOv3 asset config must resolve to a mapping.")
    missing = {
        key: value
        for key, value in payload.items()
        if str(value).startswith("MISSING_REQUIRED_LOCAL_")
    }
    if missing:
        raise FileNotFoundError(
            "Required local DINOv3 ViT-S/16 asset fields are unresolved: "
            f"{missing}. Set FASTWAM_P1_DINOV3_VITS16_WEIGHTS and "
            "FASTWAM_P1_DINOV3_VITS16_SHA256 to one verified local asset."
        )
    return DinoV3AssetSpec.from_mapping(payload)


def _visual_asset_spec(cfg: DictConfig) -> VisualBackboneAssetSpec:
    """Resolve the exact registered V2 local visual asset mapping."""

    payload = OmegaConf.to_container(cfg.p1.visual_backbone, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("V2 visual backbone config must resolve to a mapping.")
    return VisualBackboneAssetSpec.from_mapping(payload)


def is_visual_v2_config(cfg: DictConfig) -> bool:
    """Return whether this run opts into the independent V2 lineage."""

    return str(cfg.p1.get("lineage", "dino_v1")) == "visual_v2"


def visual_input_contract_sha256(cfg: DictConfig) -> str:
    """Return the lineage-specific camera-geometry contract hash."""

    field = (
        "visual_camera_input_contract_sha256"
        if is_visual_v2_config(cfg)
        else "camera_input_contract_sha256"
    )
    return str(cfg.p1[field])


def _verify_file(path: str | Path, expected: str, *, label: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is missing: {candidate}")
    actual = sha256_file(candidate)
    if actual != str(expected).lower():
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, observed {actual}."
        )
    return actual


def audit_p1_assets(cfg: DictConfig) -> dict[str, Any]:
    """Verify all declared local assets before allocating either model."""

    visual_asset = None
    if is_visual_v2_config(cfg):
        visual_asset = _visual_asset_spec(cfg)
        verify_visual_backbone_asset(visual_asset)
        asset = None
    else:
        asset = _asset_spec(cfg)
        if not asset.source_root.is_dir():
            raise FileNotFoundError(
                f"DINOv3 source root is missing: {asset.source_root}"
            )
        if not asset.weights_path.is_file():
            raise FileNotFoundError(
                "Required local DINOv3 ViT-S/16 weights are missing: "
                f"{asset.weights_path}"
            )
    dataset_info_sha256 = _verify_file(
        cfg.data.dataset_info,
        cfg.data.dataset_info_sha256,
        label="P1 dataset info",
    )
    dataset_info = json.loads(
        Path(str(cfg.data.dataset_info)).read_text(encoding="utf-8")
    )
    if dataset_info.get("codebase_version") != "v2.1":
        raise ValueError("P1 LeRobot dataset codebase revision is not v2.1.")
    result = {
        "parent_checkpoint_sha256": _verify_file(
            cfg.parent.checkpoint,
            cfg.parent.checkpoint_sha256,
            label="FastWAM parent",
        ),
        "statistics_sha256": _verify_file(
            cfg.parent.statistics,
            cfg.parent.statistics_sha256,
            label="FastWAM statistics",
        ),
        "wan_vae_sha256": _verify_file(
            cfg.p1.fastwam_runtime.vae_path,
            cfg.p1.fastwam_runtime.vae_sha256,
            label="Wan2.2 local VAE",
        ),
        "dataset_metadata_sha256": _verify_file(
            cfg.data.dataset_metadata,
            cfg.data.dataset_metadata_sha256,
            label="P1 dataset episode metadata",
        ),
        "dataset_info_sha256": dataset_info_sha256,
        "dataset_revision": str(cfg.data.dataset_revision),
    }
    if visual_asset is not None:
        result.update(
            visual_backbone=visual_asset.checkpoint_metadata(
                camera_ids=tuple(cfg.p1.camera_ids),
                input_contract_sha256=visual_input_contract_sha256(cfg),
            ),
            visual_weights_sha256=visual_asset.weights_sha256,
            visual_source_revision=visual_asset.source_revision,
        )
    else:
        assert asset is not None
        result.update(
            dinov3_weights_sha256=_verify_file(
                asset.weights_path,
                asset.weights_sha256,
                label="DINOv3 ViT-S/16 weights",
            ),
            dinov3_source_revision=asset.source_revision,
        )
    return result


def record_p1_failure(
    cfg: DictConfig,
    error: BaseException,
    *,
    traceback_text: str,
    last_metrics: Mapping[str, Any] | None = None,
) -> Path | None:
    """Preserve BLOCKED/FAIL evidence after the output directory is claimed."""

    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if not (output_dir / P1_OUTPUT_MARKER).is_file():
        return None
    blocked = isinstance(error, FileNotFoundError)
    status = "BLOCKED" if blocked else "FAIL"
    root = Path(__file__).resolve().parents[2]
    resolved = OmegaConf.to_container(cfg, resolve=True)
    failed_stage = {
        "t1_smoke": "t1",
        "tiny_overfit": "t2",
        "short_pilot": "t3",
    }[str(cfg.runner.stage)]
    stage_status = {
        "t1": "PASS" if failed_stage != "t1" else status,
        "t2": (
            "PASS"
            if failed_stage == "t3"
            else status
            if failed_stage == "t2"
            else "NOT-RUN"
        ),
        "t3": status if failed_stage == "t3" else "NOT-RUN",
        "t4": "NOT-RUN",
    }
    metrics = {
        "schema": "fastwam-p1-dino-bc-metrics-v1",
        "status": status,
        "last_valid_stage": None,
        **stage_status,
        **dict(last_metrics or {}),
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    manifest = {
        "schema": "fastwam-p1-dino-bc-run-manifest-v1",
        "status": status,
        "exit_status": 1,
        "stage": str(cfg.runner.stage),
        "arm": str(cfg.runner.arm),
        "command": list(sys.argv),
        "resolved_config_sha256": _sha256_json(resolved),
        "repositories": {
            "fastwam": _git_state(root),
            "outer": _git_state(root.parent),
        },
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback_text,
        "stage_status": stage_status,
        "reader_lora_checkpoint": {
            "path": str(output_dir / "reader_lora_checkpoint.pt"),
            "exists": (output_dir / "reader_lora_checkpoint.pt").is_file(),
            "reason_absent": (
                f"{failed_stage.upper()} did not complete; no checkpoint was fabricated."
                if not (output_dir / "reader_lora_checkpoint.pt").is_file()
                else None
            ),
        },
        "created_unix_seconds": time.time(),
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    return output_dir / "run_manifest.json"


def instantiate_p1_smoke_dataset(cfg: DictConfig):
    payload = OmegaConf.to_container(cfg.data.smoke, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("P1 smoke dataset config must resolve to a mapping.")
    split_seed = int(payload.pop("split_seed"))
    if split_seed != int(cfg.seed):
        raise ValueError("P1 smoke dataset split seed changed.")
    return instantiate(OmegaConf.create(payload))


def _reader_gradient_report(policy: FastWAMP1DinoBCPolicy) -> dict[str, Any]:
    groups = {"w_q": [], "w_o": [], "w_g": []}
    for name, parameter in policy.visual_reader.named_parameters():
        if ".query_projection." in name:
            groups["w_q"].append((name, parameter))
        elif ".output_projection." in name:
            groups["w_o"].append((name, parameter))
        elif ".semantic_gate." in name:
            groups["w_g"].append((name, parameter))
    report = {}
    for family, values in groups.items():
        gradients = [parameter.grad for _, parameter in values]
        report[family] = {
            "parameter_names": [name for name, _ in values],
            "all_present": bool(values)
            and all(value is not None for value in gradients),
            "all_finite": bool(values)
            and all(
                value is not None and torch.isfinite(value).all() for value in gradients
            ),
            "nonzero_count": int(
                sum(
                    torch.count_nonzero(value).item()
                    for value in gradients
                    if value is not None
                )
            ),
        }
    return report


def _assert_gradient_gate(report: Mapping[str, Any], families: tuple[str, ...]) -> None:
    failed = {
        family: report[family]
        for family in families
        if not report[family]["all_present"]
        or not report[family]["all_finite"]
        or report[family]["nonzero_count"] <= 0
    }
    if failed:
        raise RuntimeError(f"P1 reader gradient gate failed: {failed}.")


def build_real_p1_policy(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> tuple[FastWAMP1DinoBCPolicy, dict[str, Any]]:
    """Allocate the hash-verified frozen parents and configured P1 policy."""

    runtime = cfg.p1.fastwam_runtime
    if not bool(runtime.skip_download):
        raise ValueError("P1 local-only runtime must set skip_download=true.")
    model_base_path = Path(str(runtime.model_base_path)).expanduser().resolve()
    expected_vae = model_base_path / (
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
    )
    if Path(str(runtime.vae_path)).expanduser().resolve() != expected_vae:
        raise ValueError("P1 VAE path is outside the declared local model base.")
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    visual_v2 = is_visual_v2_config(cfg)
    input_contract_hash = visual_input_contract_sha256(cfg)
    if visual_v2:
        visual_asset = _visual_asset_spec(cfg)
        visual_encoder = build_frozen_visual_encoder(visual_asset, device=device)
    else:
        asset = _asset_spec(cfg)
        visual_encoder = FrozenDinoV3Encoder.from_local_asset(asset, device=device)
    actor = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device))
    parent_load = load_strict_fastwam_parent(actor, str(cfg.parent.checkpoint))
    if visual_v2:
        assert isinstance(visual_encoder, FrozenVisualPatchEncoder)
        memory_hash = visual_encoder.memory_contract_sha256(
            camera_ids=tuple(cfg.p1.camera_ids),
            input_contract_sha256=input_contract_hash,
        )
        reader = build_visual_patch_reader(
            VisualPatchReaderConfig(
                action_hidden_dim=int(cfg.model.action_dit_config.hidden_dim),
                timestep_dim=int(cfg.model.action_dit_config.hidden_dim),
                proprio_dim=int(cfg.bc_policy.proprio_dim),
                memory_dim=visual_encoder.asset.preset.native_dim,
                camera_ids=tuple(cfg.p1.camera_ids),
                layer_indices=tuple(
                    int(index) for index in cfg.p1.reader.layer_indices
                ),
                temperature=float(cfg.p1.reader.temperature),
                residual_scale=float(cfg.p1.reader.residual_scale),
                query_projection=ProjectionSpec.from_mapping(
                    OmegaConf.to_container(
                        cfg.p1.reader.query_projection,
                        resolve=True,
                    )
                ),
                output_projection=ProjectionSpec.from_mapping(
                    OmegaConf.to_container(
                        cfg.p1.reader.output_projection,
                        resolve=True,
                    )
                ),
                memory_contract_sha256=memory_hash,
                semantic_gate_floor=float(
                    cfg.p1.reader.get("semantic_gate_floor", 0.0)
                ),
                semantic_gate_temperature=float(
                    cfg.p1.reader.get("semantic_gate_temperature", 1.0)
                ),
            )
        )
    else:
        memory_hash = native_memory_contract_sha256(
            asset,
            camera_ids=tuple(cfg.p1.camera_ids),
            input_contract_sha256=input_contract_hash,
        )
        reader = build_dino_semantic_reader(
            DinoSemanticReaderConfig(
                action_hidden_dim=int(cfg.model.action_dit_config.hidden_dim),
                timestep_dim=int(cfg.model.action_dit_config.hidden_dim),
                proprio_dim=int(cfg.bc_policy.proprio_dim),
                camera_ids=tuple(cfg.p1.camera_ids),
                layer_indices=tuple(
                    int(index) for index in cfg.p1.reader.layer_indices
                ),
                temperature=float(cfg.p1.reader.temperature),
                residual_scale=float(cfg.p1.reader.residual_scale),
                query_projection=ProjectionSpec.from_mapping(
                    OmegaConf.to_container(
                        cfg.p1.reader.query_projection,
                        resolve=True,
                    )
                ),
                output_projection=ProjectionSpec.from_mapping(
                    OmegaConf.to_container(
                        cfg.p1.reader.output_projection,
                        resolve=True,
                    )
                ),
                memory_contract_sha256=memory_hash,
                semantic_gate_floor=float(
                    cfg.p1.reader.get("semantic_gate_floor", 0.0)
                ),
                semantic_gate_temperature=float(
                    cfg.p1.reader.get("semantic_gate_temperature", 1.0)
                ),
            )
        )
    policy_type = FastWAMP1VisualBCPolicy if visual_v2 else FastWAMP1DinoBCPolicy
    policy = policy_type(
        actor=actor,
        lora_config=RegimeLoRAConfig(**OmegaConf.to_container(cfg.lora, resolve=True)),
        visual_encoder=visual_encoder,
        visual_reader=reader,
        config=FastWAMP1DinoBCConfig(
            action=FastWAMUncondBCConfig(
                **OmegaConf.to_container(cfg.bc_policy, resolve=True)
            ),
            camera_ids=tuple(cfg.p1.camera_ids),
            camera_input_contract_sha256=input_contract_hash,
            position_mode=str(cfg.p1.position_mode),
        ),
    ).to(device)
    return policy, parent_load


def _run_t1(
    cfg: DictConfig, *, output_dir: Path, assets: Mapping[str, Any]
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("P1 real T1 requires a CUDA GPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("P1 minimal feasibility stages run on exactly one GPU.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    misc.register_work_dir(output_dir)

    dataset = instantiate_p1_smoke_dataset(cfg)
    indices = [int(index) for index in cfg.data.smoke_window_indices]
    if len(indices) < 2 or any(not 0 <= index < len(dataset) for index in indices):
        raise ValueError("P1 T1 smoke window indices are invalid.")
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        )
    )
    policy, parent_load = build_real_p1_policy(cfg, device=device)
    actor = policy.actor
    visual_encoder = policy.visual_encoder
    reader = policy.visual_reader
    memory_hash = policy.expected_memory_contract
    policy.train()
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=float(cfg.optimizer.lora_learning_rate),
        reader_learning_rate=float(cfg.optimizer.reader_learning_rate),
        train_lora=True,
        betas=tuple(float(value) for value in cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    camera_batch = policy._camera_batch(batch)
    memory = visual_encoder.encode(camera_batch)
    if memory.tokens.shape != (2, 2, 196, 384):
        raise RuntimeError(f"P1 real memory shape mismatch: {memory.tokens.shape}.")
    if memory.tokens.is_inference() or memory.tokens.requires_grad:
        raise RuntimeError("P1 real memory is not an ordinary detached tensor.")
    action = batch["action"].to(device=device, dtype=policy.dtype)
    timestep = torch.tensor([200.0, 700.0], device=device, dtype=policy.dtype)
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1)
    noise = torch.randn(
        action.shape, generator=generator, device=device, dtype=policy.dtype
    )
    noisy_action = actor.train_action_scheduler.add_noise(action, noise, timestep)
    dino_calls = {"count": 0}

    def count_dino(_module, _inputs, _outputs) -> None:
        dino_calls["count"] += 1

    hook = visual_encoder.register_forward_hook(count_dino)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sidecar_off = policy.predict_velocity(
                batch,
                noisy_action=noisy_action,
                timestep=timestep,
                include_visual=False,
            )
            zero_a3 = policy.predict_velocity(
                batch,
                noisy_action=noisy_action,
                timestep=timestep,
                include_visual=True,
            )
            calls_before_idm = dino_calls["count"]
            policy.predict_velocity(
                batch,
                noisy_action=noisy_action,
                timestep=timestep,
                regime="idm",
                include_visual=False,
            )
            idm_dino_calls = dino_calls["count"] - calls_before_idm
    finally:
        hook.remove()
    if not torch.equal(sidecar_off, zero_a3):
        raise RuntimeError("P1 zero-A3 no longer exactly matches sidecar-off.")
    if idm_dino_calls != 0:
        raise RuntimeError("IDM unexpectedly called the P1 DINO sidecar.")

    step_reports = []
    last_loss = None
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = policy(batch, timestep=timestep, noise=noise)
            loss = output["loss_action_bc"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"P1 T1 step {step + 1} loss is non-finite.")
        loss.backward()
        gradient_report = _reader_gradient_report(policy)
        _assert_gradient_gate(
            gradient_report,
            ("w_o",) if step == 0 else ("w_q", "w_g"),
        )
        if any(parameter.grad is not None for parameter in visual_encoder.parameters()):
            raise RuntimeError("P1 frozen DINO received gradients.")
        lora_and_reader = {
            id(parameter)
            for values in policy.parameter_families().values()
            for parameter in values
        }
        frozen_with_grad = [
            name
            for name, parameter in policy.named_parameters()
            if id(parameter) not in lora_and_reader and parameter.grad is not None
        ]
        if frozen_with_grad:
            raise RuntimeError(
                f"P1 frozen base received gradients: {frozen_with_grad[:16]}."
            )
        optimizer.step()
        last_loss = float(loss.detach().float().item())
        step_reports.append(
            {
                "step": step + 1,
                "loss_action_bc": last_loss,
                "reader_gradients": gradient_report,
            }
        )

    resolved = OmegaConf.to_container(cfg, resolve=True)
    contract = {
        "resolved_config_sha256": _sha256_json(resolved),
        "dataset_metadata_sha256": assets["dataset_metadata_sha256"],
        "camera_input_contract_sha256": visual_input_contract_sha256(cfg),
        "memory_contract_sha256": memory_hash,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "layout": memory.layout_contract,
        "spatial_pairs": OmegaConf.to_container(cfg.p1.spatial_pairs, resolve=True),
    }
    checkpoint = output_dir / "reader_lora_checkpoint.pt"
    save_p1_dino_bc_checkpoint(
        checkpoint,
        adapter=policy.lora_adapter,
        reader=reader,
        global_step=2,
        stage="t1_smoke",
        arm="a3_joint",
        parent_checkpoint_sha256=assets["parent_checkpoint_sha256"],
        dinov3_weights_sha256=assets["dinov3_weights_sha256"],
        memory_contract_sha256=memory_hash,
        contract=contract,
        provenance={
            "statistics_sha256": assets["statistics_sha256"],
            "dinov3_source_revision": assets["dinov3_source_revision"],
            "parent_load": parent_load,
            "smoke_window_indices": indices,
        },
        trainer_state={
            "last_loss_action_bc": last_loss,
            "best_dev_loss_action_bc": None,
            "nonzero_update_count": 2,
        },
    )
    checkpoint_report = inspect_p1_dino_bc_checkpoint(checkpoint)
    metrics = {
        "schema": "fastwam-p1-dino-bc-metrics-v1",
        "status": "PASS",
        "t1": "PASS",
        "t2": "NOT-RUN",
        "t3": "NOT-RUN",
        "t4": "NOT-RUN",
        "memory_shape": list(memory.tokens.shape),
        "memory_is_inference": memory.tokens.is_inference(),
        "memory_requires_grad": memory.tokens.requires_grad,
        "layout": memory.layout_contract,
        "memory_contract_sha256": memory.memory_contract_sha256,
        "sidecar_off_zero_a3_exact": True,
        "idm_dino_calls": idm_dino_calls,
        "optimizer_steps": step_reports,
        "checkpoint_inspection": checkpoint_report,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    return metrics


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _resolve_spatial_pair_indices(dataset, cfg: DictConfig) -> list[dict[str, Any]]:
    """Resolve preregistered episode/frame identities to stable dataset indices."""

    lerobot = dataset.lerobot_dataset.multi_dataset
    if len(lerobot._datasets) != 1:
        raise ValueError("P1 tiny spatial fixture must use exactly one LIBERO suite.")
    source = lerobot._datasets[0]
    episodes = source.hf_dataset["episode_index"]
    frames = source.hf_dataset["frame_index"]
    identity_to_index = {
        (_as_int(episode), _as_int(frame)): index
        for index, (episode, frame) in enumerate(zip(episodes, frames, strict=True))
    }
    resolved = []
    for pair in OmegaConf.to_container(cfg.p1.spatial_pairs, resolve=True):
        endpoints = []
        for side in ("left", "right"):
            identity = (
                int(pair[side]["episode_index"]),
                int(pair[side]["frame_index"]),
            )
            if identity not in identity_to_index:
                raise ValueError(
                    f"P1 spatial endpoint {identity} is absent from the dataset."
                )
            endpoints.append(identity_to_index[identity])
        resolved.append({**pair, "dataset_indices": endpoints})
    if len({index for pair in resolved for index in pair["dataset_indices"]}) != 8:
        raise ValueError("P1 tiny fixture must resolve to eight unique windows.")
    return resolved


def _slice_batch_value(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, Mapping):
        return {
            key: _slice_batch_value(item, index, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_slice_batch_value(item, index, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, index, batch_size) for item in value)
    return value


def _slice_memory(
    memory: NativePatchMemory | SpatialPatchMemory,
    index: int,
) -> NativePatchMemory | SpatialPatchMemory:
    return replace(
        memory,
        tokens=memory.tokens[index : index + 1],
        patch_valid_mask=memory.patch_valid_mask[index : index + 1],
        camera_valid_mask=memory.camera_valid_mask[index : index + 1],
    )


def _slice_condition(
    condition: CachedActionCondition,
    index: int,
) -> CachedActionCondition:
    if condition.visual is None:
        raise ValueError("P1 tiny condition is missing visual memory.")
    batch_size = condition.context.shape[0]
    return replace(
        condition,
        context=condition.context[index : index + 1],
        context_mask=condition.context_mask[index : index + 1],
        video_kv_cache=_slice_batch_value(
            condition.video_kv_cache,
            index,
            batch_size,
        ),
        visual=replace(
            condition.visual,
            memory=_slice_memory(condition.visual.memory, index),
            proprio=condition.visual.proprio[index : index + 1],
        ),
    )


def _slice_batch(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    batch_size = batch["action"].shape[0]
    return {
        key: _slice_batch_value(value, index, batch_size)
        for key, value in batch.items()
    }


def _reader_update_report(
    policy: FastWAMP1DinoBCPolicy,
    before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    groups = {"w_q": [], "w_o": [], "w_g": []}
    for name, parameter in policy.visual_reader.named_parameters():
        if ".query_projection." in name:
            family = "w_q"
        elif ".output_projection." in name:
            family = "w_o"
        elif ".semantic_gate." in name:
            family = "w_g"
        else:
            continue
        delta = parameter.detach().float().cpu() - before[name].float()
        groups[family].append(delta)
    return {
        family: {
            "finite": bool(values)
            and all(torch.isfinite(value).all() for value in values),
            "update_norm": float(
                torch.stack([value.square().sum() for value in values]).sum().sqrt()
            )
            if values
            else 0.0,
        }
        for family, values in groups.items()
    }


@torch.no_grad()
def _evaluate_tiny_cache(
    policy: FastWAMP1DinoBCPolicy,
    cached_pairs: list[dict[str, Any]],
    *,
    memory_mode: str,
) -> float:
    total = 0.0
    samples = 0
    for record in cached_pairs:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = policy.loss_from_prepared_condition(
                record["batch"],
                condition=record["condition"],
                timestep=record["timestep"],
                noise=record["noise"],
                memory_mode=memory_mode,
            )
        loss = output["loss_action_bc"].detach().float()
        if not torch.isfinite(loss):
            raise RuntimeError(f"P1 tiny {memory_mode} loss is non-finite.")
        batch_size = record["batch"]["action"].shape[0]
        total += float(loss.item()) * batch_size
        samples += batch_size
    return total / samples


@torch.no_grad()
def _evaluate_spatial_pairs(
    policy: FastWAMP1DinoBCPolicy,
    cached_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_reports = []
    directional_relative_deltas = []
    positive_pair_count = 0
    for record in cached_pairs:
        directions = []
        memory = record["condition"].visual.memory
        for source_index, memory_index in ((0, 1), (1, 0)):
            sample_batch = _slice_batch(record["batch"], source_index)
            sample_condition = _slice_condition(record["condition"], source_index)
            kwargs = {
                "batch": sample_batch,
                "condition": sample_condition,
                "timestep": record["timestep"][source_index : source_index + 1],
                "noise": record["noise"][source_index : source_index + 1],
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                correct = (
                    policy.loss_from_prepared_condition(
                        **kwargs,
                        memory_mode="correct",
                    )["loss_action_bc"]
                    .detach()
                    .float()
                )
                swapped = (
                    policy.loss_from_prepared_condition(
                        **kwargs,
                        memory_mode="correct",
                        memory_override=_slice_memory(memory, memory_index),
                    )["loss_action_bc"]
                    .detach()
                    .float()
                )
            relative = float(
                (swapped - correct).item() / max(abs(correct.item()), 1e-12)
            )
            directional_relative_deltas.append(relative)
            directions.append(
                {
                    "source_side": "left" if source_index == 0 else "right",
                    "memory_side": "right" if memory_index == 1 else "left",
                    "correct_loss": float(correct.item()),
                    "swapped_loss": float(swapped.item()),
                    "relative_delta": relative,
                }
            )
        bidirectional_mean = sum(value["relative_delta"] for value in directions) / 2.0
        if bidirectional_mean > 0:
            positive_pair_count += 1
        pair_reports.append(
            {
                "pair_id": record["pair"]["pair_id"],
                "directions": directions,
                "bidirectional_mean_relative_delta": bidirectional_mean,
            }
        )
    return {
        "pairs": pair_reports,
        "median_directional_relative_delta": median(directional_relative_deltas),
        "positive_bidirectional_pair_count": positive_pair_count,
        "pair_count": len(pair_reports),
    }


def _run_t2(
    cfg: DictConfig,
    *,
    output_dir: Path,
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("P1 real T2 requires a CUDA GPU.")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    misc.register_work_dir(output_dir)
    dataset = instantiate_p1_smoke_dataset(cfg)
    resolved_pairs = _resolve_spatial_pair_indices(dataset, cfg)
    policy, parent_load = build_real_p1_policy(cfg, device=device)
    policy.train()
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=float(cfg.optimizer.lora_learning_rate),
        reader_learning_rate=float(cfg.optimizer.reader_learning_rate),
        train_lora=False,
        betas=tuple(float(value) for value in cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    if any(
        parameter.requires_grad for parameter in policy.lora_adapter.lora_parameters()
    ):
        raise RuntimeError("P1 T2 must freeze every LoRA parameter.")
    cached_pairs = []
    for pair in resolved_pairs:
        batch = next(
            iter(
                DataLoader(
                    Subset(dataset, pair["dataset_indices"]),
                    batch_size=2,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
            )
        )
        condition = policy.prepare_action_condition(batch, include_visual=True)
        identities = [f"{pair['pair_id']}:{side}" for side in ("left", "right")]
        timestep, noise = stateless_validation_flow_inputs(
            sample_identities=identities,
            action_shape=tuple(batch["action"].shape),
            scheduler=policy.actor.train_action_scheduler,
            seed=int(cfg.seed),
            device=device,
            dtype=policy.dtype,
        )
        cached_pairs.append(
            {
                "pair": pair,
                "batch": batch,
                "condition": condition,
                "timestep": timestep,
                "noise": noise,
            }
        )

    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.visual_reader.named_parameters()
    }
    initial_loss = _evaluate_tiny_cache(policy, cached_pairs, memory_mode="correct")
    steps = int(cfg.training.tiny_steps)
    train_loss_trace = []
    nonzero_updates = 0
    reader_parameters = tuple(policy.visual_reader.parameters())
    for step in range(steps):
        record = cached_pairs[step % len(cached_pairs)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = policy.loss_from_prepared_condition(
                record["batch"],
                condition=record["condition"],
                timestep=record["timestep"],
                noise=record["noise"],
                memory_mode="correct",
            )
            loss = output["loss_action_bc"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"P1 T2 step {step + 1} loss is non-finite.")
        loss.backward()
        gradients = [
            parameter.grad.detach().float()
            for parameter in reader_parameters
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"P1 T2 step {step + 1} reader gradients are invalid.")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            reader_parameters,
            float(cfg.optimizer.gradient_clip),
        )
        optimizer.step()
        if float(gradient_norm) > 0:
            nonzero_updates += 1
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            train_loss_trace.append(
                {"step": step + 1, "loss_action_bc": float(loss.detach().float())}
            )

    correct_loss = _evaluate_tiny_cache(policy, cached_pairs, memory_mode="correct")
    zero_loss = _evaluate_tiny_cache(policy, cached_pairs, memory_mode="zero")
    shuffled_loss = _evaluate_tiny_cache(policy, cached_pairs, memory_mode="shuffled")
    spatial = _evaluate_spatial_pairs(policy, cached_pairs)
    updates = _reader_update_report(policy, before)
    final_ratio = correct_loss / max(initial_loss, 1e-12)
    zero_improvement = (zero_loss - correct_loss) / max(abs(zero_loss), 1e-12)
    shuffled_improvement = (shuffled_loss - correct_loss) / max(
        abs(shuffled_loss),
        1e-12,
    )
    thresholds = cfg.p1.thresholds
    gates = {
        "loss_reduction": final_ratio <= float(thresholds.tiny_final_over_initial_max),
        "memory_dependence": max(zero_improvement, shuffled_improvement)
        >= float(thresholds.memory_relative_improvement_min),
        "spatial_median": spatial["median_directional_relative_delta"]
        >= float(thresholds.delta_position_relative_loss),
        "spatial_pair_count": spatial["positive_bidirectional_pair_count"]
        >= int(thresholds.spatial_positive_pairs_min),
        "reader_updates": all(
            report["finite"] and report["update_norm"] > 0
            for report in updates.values()
        ),
        "lora_frozen": all(
            not parameter.requires_grad
            for parameter in policy.lora_adapter.lora_parameters()
        ),
    }
    overall_learnable = gates["loss_reduction"] and gates["memory_dependence"]
    spatial_pass = gates["spatial_median"] and gates["spatial_pair_count"]
    if overall_learnable and spatial_pass:
        position_decision = "GO-NATIVE-POS"
    elif overall_learnable:
        position_decision = "GO-COORD-SCORE-EXPERIMENT"
    else:
        position_decision = "NO-GO-POS"
    status = "PASS" if all(gates.values()) else "FAIL"

    resolved = OmegaConf.to_container(cfg, resolve=True)
    contract = {
        "resolved_config_sha256": _sha256_json(resolved),
        "dataset_metadata_sha256": assets["dataset_metadata_sha256"],
        "camera_input_contract_sha256": visual_input_contract_sha256(cfg),
        "memory_contract_sha256": policy.expected_memory_contract,
        "reader_contract_sha256": policy.visual_reader.reader_contract_sha256,
        "layout": policy.p1_config.layout_contract,
        "spatial_pairs": resolved_pairs,
    }
    checkpoint = output_dir / "reader_lora_checkpoint.pt"
    save_p1_dino_bc_checkpoint(
        checkpoint,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        global_step=steps,
        stage="tiny_overfit",
        arm="a3_joint",
        parent_checkpoint_sha256=assets["parent_checkpoint_sha256"],
        dinov3_weights_sha256=assets["dinov3_weights_sha256"],
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={
            "statistics_sha256": assets["statistics_sha256"],
            "dinov3_source_revision": assets["dinov3_source_revision"],
            "parent_load": parent_load,
            "tiny_window_count": 8,
            "lora_frozen": True,
        },
        trainer_state={
            "last_loss_action_bc": correct_loss,
            "best_dev_loss_action_bc": None,
            "nonzero_update_count": nonzero_updates,
        },
    )
    checkpoint_report = inspect_p1_dino_bc_checkpoint(checkpoint)
    metrics = {
        "schema": "fastwam-p1-dino-bc-metrics-v1",
        "status": status,
        "t1": "PASS",
        "t2": status,
        "t3": "NOT-RUN",
        "t4": "NOT-RUN",
        "optimizer_steps": steps,
        "nonzero_update_count": nonzero_updates,
        "initial_correct_loss": initial_loss,
        "final_correct_loss": correct_loss,
        "final_over_initial": final_ratio,
        "zero_memory_loss": zero_loss,
        "shuffled_memory_loss": shuffled_loss,
        "correct_vs_zero_relative_improvement": zero_improvement,
        "correct_vs_shuffled_relative_improvement": shuffled_improvement,
        "train_loss_trace": train_loss_trace,
        "reader_updates": updates,
        "spatial": spatial,
        "gates": gates,
        "position_decision": position_decision,
        "resolved_spatial_pairs": resolved_pairs,
        "checkpoint_inspection": checkpoint_report,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    return metrics


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolve_episode_frame_indices(
    dataset,
    *,
    episode_ids: list[int],
    frame_index: int,
) -> list[dict[str, int]]:
    lerobot = dataset.lerobot_dataset.multi_dataset
    if len(lerobot._datasets) != 1:
        raise ValueError("P1 short fixture must use exactly one LIBERO suite.")
    source = lerobot._datasets[0]
    identities = {
        (_as_int(episode), _as_int(frame)): index
        for index, (episode, frame) in enumerate(
            zip(
                source.hf_dataset["episode_index"],
                source.hf_dataset["frame_index"],
                strict=True,
            )
        )
    }
    resolved = []
    for episode_id in episode_ids:
        identity = (int(episode_id), int(frame_index))
        if identity not in identities:
            raise ValueError(f"P1 short window {identity} is absent from the dataset.")
        resolved.append(
            {
                "episode_index": identity[0],
                "frame_index": identity[1],
                "dataset_index": identities[identity],
            }
        )
    return resolved


def _prepare_short_records(
    policy: FastWAMP1DinoBCPolicy,
    dataset,
    identities: list[dict[str, int]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    records = []
    for offset in range(0, len(identities), 2):
        selected = identities[offset : offset + 2]
        batch = next(
            iter(
                DataLoader(
                    Subset(
                        dataset,
                        [item["dataset_index"] for item in selected],
                    ),
                    batch_size=len(selected),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
            )
        )
        condition_a3 = policy.prepare_action_condition(batch, include_visual=True)
        condition_a0 = replace(condition_a3, visual=None)
        sample_identities = [
            f"episode={item['episode_index']}:frame={item['frame_index']}"
            for item in selected
        ]
        timestep, noise = stateless_validation_flow_inputs(
            sample_identities=sample_identities,
            action_shape=tuple(batch["action"].shape),
            scheduler=policy.actor.train_action_scheduler,
            seed=seed,
            device=policy.device,
            dtype=policy.dtype,
        )
        records.append(
            {
                "identities": selected,
                "batch": batch,
                "condition_a0": condition_a0,
                "condition_a3": condition_a3,
                "timestep": timestep,
                "noise": noise,
            }
        )
    return records


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )


def _train_short_arm(
    policy: FastWAMP1DinoBCPolicy,
    records: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    steps: int,
    gradient_clip: float,
) -> dict[str, Any]:
    condition_key = "condition_a0" if arm == "a0_bc" else "condition_a3"
    memory_mode = "off" if arm == "a0_bc" else "correct"
    parameters = _optimizer_parameters(optimizer)
    trace = []
    nonzero_updates = 0
    last_loss = None
    for step in range(steps):
        record = records[step % len(records)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = policy.loss_from_prepared_condition(
                record["batch"],
                condition=record[condition_key],
                timestep=record["timestep"],
                noise=record["noise"],
                memory_mode=memory_mode,
            )["loss_action_bc"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"P1 T3 {arm} step {step + 1} loss is non-finite.")
        loss.backward()
        gradients = [
            parameter.grad.detach().float()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"P1 T3 {arm} gradients are absent or non-finite.")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        optimizer.step()
        if float(gradient_norm) > 0:
            nonzero_updates += 1
        last_loss = float(loss.detach().float().item())
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            trace.append({"step": step + 1, "loss_action_bc": last_loss})
    return {
        "last_loss_action_bc": last_loss,
        "nonzero_update_count": nonzero_updates,
        "train_loss_trace": trace,
    }


@torch.no_grad()
def _evaluate_short_arm(
    policy: FastWAMP1DinoBCPolicy,
    records: list[dict[str, Any]],
    *,
    arm: str,
    memory_mode: str,
) -> float:
    condition_key = "condition_a0" if arm == "a0_bc" else "condition_a3"
    total = 0.0
    samples = 0
    for record in records:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (
                policy.loss_from_prepared_condition(
                    record["batch"],
                    condition=record[condition_key],
                    timestep=record["timestep"],
                    noise=record["noise"],
                    memory_mode=memory_mode,
                )["loss_action_bc"]
                .detach()
                .float()
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"P1 T3 {arm}/{memory_mode} dev loss is non-finite.")
        batch_size = record["batch"]["action"].shape[0]
        total += float(loss.item()) * batch_size
        samples += batch_size
    return total / samples


def _run_t3(
    cfg: DictConfig,
    *,
    output_dir: Path,
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("P1 real T3 requires a CUDA GPU.")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    misc.register_work_dir(output_dir)
    dataset = instantiate_p1_smoke_dataset(cfg)
    frame_index = int(cfg.data.short_frame_index)
    train_windows = _resolve_episode_frame_indices(
        dataset,
        episode_ids=[int(value) for value in cfg.data.short_train_episode_ids],
        frame_index=frame_index,
    )
    dev_windows = _resolve_episode_frame_indices(
        dataset,
        episode_ids=[int(value) for value in cfg.data.short_dev_episode_ids],
        frame_index=frame_index,
    )
    train_episodes = {item["episode_index"] for item in train_windows}
    dev_episodes = {item["episode_index"] for item in dev_windows}
    if train_episodes & dev_episodes:
        raise RuntimeError("P1 T3 resolved train/dev episodes overlap.")

    policy, parent_load = build_real_p1_policy(cfg, device=device)
    policy.train()
    train_records = _prepare_short_records(
        policy,
        dataset,
        train_windows,
        seed=int(cfg.seed) + 1000,
    )
    dev_records = _prepare_short_records(
        policy,
        dataset,
        dev_windows,
        seed=int(cfg.seed) + 2000,
    )
    initial_lora = policy.lora_adapter.lora_state_dict()
    initial_reader = {
        name: value.detach().cpu().clone()
        for name, value in policy.visual_reader.state_dict().items()
    }
    initial_lora_sha256 = _tensor_state_sha256(initial_lora)
    initial_reader_sha256 = _tensor_state_sha256(initial_reader)
    steps = int(cfg.training.short_steps)
    optimizer_kwargs = {
        "lora_learning_rate": float(cfg.optimizer.lora_learning_rate),
        "reader_learning_rate": float(cfg.optimizer.reader_learning_rate),
        "betas": tuple(float(value) for value in cfg.optimizer.betas),
        "eps": float(cfg.optimizer.eps),
        "weight_decay": float(cfg.optimizer.weight_decay),
    }

    a0_optimizer = build_p1_optimizer(
        policy,
        train_lora=True,
        train_reader=False,
        **optimizer_kwargs,
    )
    a0_train = _train_short_arm(
        policy,
        train_records,
        a0_optimizer,
        arm="a0_bc",
        steps=steps,
        gradient_clip=float(cfg.optimizer.gradient_clip),
    )
    a0_dev = _evaluate_short_arm(
        policy,
        dev_records,
        arm="a0_bc",
        memory_mode="off",
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    split_contract = {
        "kind": str(cfg.data.split_contract),
        "frame_index": frame_index,
        "train_windows": train_windows,
        "dev_windows": dev_windows,
    }
    contract = {
        "resolved_config_sha256": _sha256_json(resolved),
        "dataset_metadata_sha256": assets["dataset_metadata_sha256"],
        "camera_input_contract_sha256": visual_input_contract_sha256(cfg),
        "memory_contract_sha256": policy.expected_memory_contract,
        "reader_contract_sha256": policy.visual_reader.reader_contract_sha256,
        "layout": policy.p1_config.layout_contract,
        "split": split_contract,
    }
    # A0 never optimizes or reads the reader, but the strict P1 artifact schema
    # still records its matched zero-initialized reader for an auditable pair.
    policy.visual_reader.requires_grad_(True)
    a0_checkpoint = output_dir / "a0_lora_checkpoint.pt"
    save_p1_dino_bc_checkpoint(
        a0_checkpoint,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        global_step=steps,
        stage="short_pilot",
        arm="a0_bc",
        parent_checkpoint_sha256=assets["parent_checkpoint_sha256"],
        dinov3_weights_sha256=assets["dinov3_weights_sha256"],
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={
            "statistics_sha256": assets["statistics_sha256"],
            "dinov3_source_revision": assets["dinov3_source_revision"],
            "parent_load": parent_load,
            "reader_trained": False,
            "split": split_contract,
        },
        trainer_state={
            "last_loss_action_bc": a0_train["last_loss_action_bc"],
            "best_dev_loss_action_bc": a0_dev,
            "nonzero_update_count": a0_train["nonzero_update_count"],
        },
    )
    a0_checkpoint_report = inspect_p1_dino_bc_checkpoint(a0_checkpoint)

    policy.lora_adapter.load_lora_state_dict(initial_lora, strict=True)
    policy.visual_reader.load_state_dict(initial_reader, strict=True)
    policy.visual_reader.requires_grad_(True)
    restored_lora_sha256 = _tensor_state_sha256(policy.lora_adapter.lora_state_dict())
    restored_reader_sha256 = _tensor_state_sha256(
        {
            name: value.detach().cpu()
            for name, value in policy.visual_reader.state_dict().items()
        }
    )
    same_initialization = (
        restored_lora_sha256 == initial_lora_sha256
        and restored_reader_sha256 == initial_reader_sha256
    )
    if not same_initialization:
        raise RuntimeError("P1 T3 failed to restore the matched A0/A3 initialization.")

    reader_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.visual_reader.named_parameters()
    }
    a3_optimizer = build_p1_optimizer(
        policy,
        train_lora=True,
        train_reader=True,
        **optimizer_kwargs,
    )
    a3_train = _train_short_arm(
        policy,
        train_records,
        a3_optimizer,
        arm="a3_joint",
        steps=steps,
        gradient_clip=float(cfg.optimizer.gradient_clip),
    )
    a3_correct_dev = _evaluate_short_arm(
        policy,
        dev_records,
        arm="a3_joint",
        memory_mode="correct",
    )
    a3_shuffled_dev = _evaluate_short_arm(
        policy,
        dev_records,
        arm="a3_joint",
        memory_mode="shuffled",
    )
    reader_updates = _reader_update_report(policy, reader_before)
    a3_over_a0 = a3_correct_dev / max(a0_dev, 1e-12)
    shuffled_improvement = (a3_shuffled_dev - a3_correct_dev) / max(
        abs(a3_shuffled_dev),
        1e-12,
    )
    thresholds = cfg.p1.thresholds
    gates = {
        "finite_losses": all(
            torch.isfinite(torch.tensor(value))
            for value in (a0_dev, a3_correct_dev, a3_shuffled_dev)
        ),
        "matched_initialization": same_initialization,
        "matched_budget": (
            a0_train["nonzero_update_count"] == steps
            and a3_train["nonzero_update_count"] == steps
        ),
        "episode_disjoint": not bool(train_episodes & dev_episodes),
        "a3_not_degraded": a3_over_a0 <= float(thresholds.short_a3_over_a0_dev_max),
        "memory_dependence": shuffled_improvement
        >= float(thresholds.short_correct_over_shuffled_improvement_min),
        "reader_updates": all(
            report["finite"] and report["update_norm"] > 0
            for report in reader_updates.values()
        ),
        "frozen_parents": (
            all(
                parameter.grad is None
                for parameter in policy.visual_encoder.parameters()
            )
            and all(
                parameter.grad is None
                for name, parameter in policy.named_parameters()
                if not (
                    ".lora_A" in name
                    or ".lora_B" in name
                    or name.startswith("visual_reader.")
                )
            )
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    checkpoint = output_dir / "reader_lora_checkpoint.pt"
    save_p1_dino_bc_checkpoint(
        checkpoint,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        global_step=steps,
        stage="short_pilot",
        arm="a3_joint",
        parent_checkpoint_sha256=assets["parent_checkpoint_sha256"],
        dinov3_weights_sha256=assets["dinov3_weights_sha256"],
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={
            "statistics_sha256": assets["statistics_sha256"],
            "dinov3_source_revision": assets["dinov3_source_revision"],
            "parent_load": parent_load,
            "reader_trained": True,
            "split": split_contract,
            "matched_a0_checkpoint_sha256": a0_checkpoint_report["checkpoint_sha256"],
        },
        trainer_state={
            "last_loss_action_bc": a3_train["last_loss_action_bc"],
            "best_dev_loss_action_bc": a3_correct_dev,
            "nonzero_update_count": a3_train["nonzero_update_count"],
        },
    )
    checkpoint_report = inspect_p1_dino_bc_checkpoint(checkpoint)
    metrics = {
        "schema": "fastwam-p1-dino-bc-metrics-v1",
        "status": status,
        "t1": "PASS",
        "t2": "PASS",
        "t3": status,
        "t4": "NOT-RUN",
        "sample_budget_per_arm": steps * 2,
        "optimizer_steps_per_arm": steps,
        "split": split_contract,
        "matched_initial_lora_sha256": initial_lora_sha256,
        "matched_initial_reader_sha256": initial_reader_sha256,
        "a0": {**a0_train, "dev_loss_action_bc": a0_dev},
        "a3": {
            **a3_train,
            "correct_dev_loss_action_bc": a3_correct_dev,
            "shuffled_dev_loss_action_bc": a3_shuffled_dev,
            "correct_over_a0": a3_over_a0,
            "correct_vs_shuffled_relative_improvement": shuffled_improvement,
            "reader_updates": reader_updates,
        },
        "gates": gates,
        "a0_checkpoint_inspection": a0_checkpoint_report,
        "checkpoint_inspection": checkpoint_report,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    return metrics


def run_p1_dino_bc(cfg: DictConfig) -> dict[str, Any]:
    """Run the next authorized P1 stage, stopping at its prerequisite gate."""

    validate_p1_config(cfg)
    output_dir = claim_p1_output(cfg)
    assets = audit_p1_assets(cfg)
    stage = str(cfg.runner.stage)
    prerequisite_path = None
    prerequisite_sha256 = None
    prerequisite_t2_path = None
    prerequisite_t2_sha256 = None
    if stage != "t1_smoke":
        prerequisite = cfg.runner.get("prerequisite_t1_manifest")
        if not prerequisite:
            raise FileNotFoundError(
                "T2/T3 require an explicit PASS T1 manifest; none was supplied."
            )
        prerequisite_path = Path(str(prerequisite)).expanduser().resolve()
        if not prerequisite_path.is_file():
            raise FileNotFoundError(
                f"P1 T1 prerequisite manifest is missing: {prerequisite_path}"
            )
        prerequisite_sha256 = sha256_file(prerequisite_path)
        payload = json.loads(prerequisite_path.read_text(encoding="utf-8"))
        if payload.get("stage_status", {}).get("t1") != "PASS":
            raise RuntimeError("The supplied P1 T1 prerequisite is not PASS.")
    if stage == "short_pilot":
        prerequisite_t2 = cfg.runner.get("prerequisite_t2_manifest")
        if not prerequisite_t2:
            raise FileNotFoundError(
                "T3 requires an explicit PASS T2 manifest; none was supplied."
            )
        prerequisite_t2_path = Path(str(prerequisite_t2)).expanduser().resolve()
        if not prerequisite_t2_path.is_file():
            raise FileNotFoundError(
                f"P1 T2 prerequisite manifest is missing: {prerequisite_t2_path}"
            )
        prerequisite_t2_sha256 = sha256_file(prerequisite_t2_path)
        t2_payload = json.loads(prerequisite_t2_path.read_text(encoding="utf-8"))
        if t2_payload.get("stage_status", {}).get("t2") != "PASS":
            raise RuntimeError("The supplied P1 T2 prerequisite is not PASS.")
    if stage == "t1_smoke":
        metrics = _run_t1(cfg, output_dir=output_dir, assets=assets)
    elif stage == "tiny_overfit":
        metrics = _run_t2(cfg, output_dir=output_dir, assets=assets)
    else:
        metrics = _run_t3(cfg, output_dir=output_dir, assets=assets)
    root = Path(__file__).resolve().parents[2]
    stage_status = {name: metrics[name] for name in ("t1", "t2", "t3", "t4")}
    manifest = {
        "schema": "fastwam-p1-dino-bc-run-manifest-v1",
        "status": metrics["status"],
        "exit_status": 0,
        "stage": stage,
        "arm": str(cfg.runner.arm),
        "command": list(sys.argv),
        "stage_status": stage_status,
        "repositories": {
            "fastwam": _git_state(root),
            "outer": _git_state(root.parent),
        },
        "assets": dict(assets),
        "prerequisite_t1_manifest": (
            {
                "path": str(prerequisite_path),
                "sha256": prerequisite_sha256,
            }
            if prerequisite_path is not None
            else None
        ),
        "prerequisite_t2_manifest": (
            {
                "path": str(prerequisite_t2_path),
                "sha256": prerequisite_t2_sha256,
            }
            if prerequisite_t2_path is not None
            else None
        ),
        "a0_lora_checkpoint": metrics.get("a0_checkpoint_inspection"),
        "reader_lora_checkpoint": metrics["checkpoint_inspection"],
        "created_unix_seconds": time.time(),
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)
    return metrics
