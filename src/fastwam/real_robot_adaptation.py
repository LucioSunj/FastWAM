"""Real-data config preparation and offline BC using existing FastWAM losses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.adapters import RegimeLoRAConfig, sha256_file
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.real_robot_tiny import TinyFastWAM, adapt_tiny_fixture
from fastwam.uncond_bc import FastWAMUncondBCConfig, FastWAMUncondBCPolicy
from fastwam.uncond_bc_trainer import load_strict_fastwam_parent


def write_adaptation_configs(dataset: Path):
    """Write a native train.py profile with unresolved real assets made explicit."""
    config_root = Path(__file__).parents[2] / "configs"
    cfg = OmegaConf.load(config_root / "train.yaml")
    del cfg["defaults"]
    task_cfg = OmegaConf.load(config_root / "task/real_robot_idm.yaml")
    cfg.merge_with(task_cfg)
    for key in task_cfg:
        if OmegaConf.is_missing(task_cfg, key):
            cfg[key] = "???"
    cfg.model = OmegaConf.load(config_root / "model/fastwam_idm.yaml")
    cfg.data = OmegaConf.load(dataset / "data_idm.yaml")
    cfg.model.model_id = "???"
    cfg.model.tokenizer_model_id = "???"
    cfg.model.action_dit_pretrained_path = "???"
    cfg.resume = "???"
    cfg.output_dir = str(dataset / "adapted_idm")
    OmegaConf.save(cfg, dataset / "idm_train.yaml")
    OmegaConf.save(
        OmegaConf.create(
            {
                "kind": "fastwam",
                "dataset": str(dataset),
                "parent_checkpoint": "???",
                "model_config": "???",
                "device": "???",
                "precision": "???",
                "output_dir": str(dataset / "uncond_bc"),
                "max_steps": "???",
                "micro_batch_size": 1,
                "learning_rate": "???",
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "seed": 42,
            }
        ),
        dataset / "uncond_bc_train.yaml",
    )


def _tiny_text_cache(dataset, actor):
    """Use the reader's existing cache naming for explicitly synthetic embeddings."""
    cache = dataset / "text_cache"
    cache.mkdir(exist_ok=True)
    tasks = set()
    for path in dataset.glob("*/meta/tasks.jsonl"):
        for line in path.read_text().splitlines():
            task = json.loads(line)["task"]
            tasks.add(task[0] if isinstance(task, list) else task)
    for task in tasks:
        prompt = DEFAULT_PROMPT.format(task=task)
        context, mask = actor.encode_prompt([prompt])
        padded, padded_mask = (
            torch.zeros(128, actor.text_dim, dtype=torch.bfloat16),
            torch.zeros(128, dtype=torch.bool),
        )
        padded[: context.shape[1]], padded_mask[: mask.shape[1]] = (
            context[0].detach(),
            mask[0],
        )
        # This is the native text-cache key, not a new artifact fingerprint.
        path = cache / (
            hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
        )
        if path.exists():
            existing = torch.load(path, weights_only=True, map_location="cpu")
            if not torch.equal(existing["context"], padded):
                raise ValueError(
                    "Tiny fixture refuses to overwrite an existing different text cache."
                )
        else:
            torch.save({"context": padded, "mask": padded_mask}, path)


def run_tiny_adaptation(dataset, output, *, steps=8, seed=42):
    """Train the tiny parent and LoRA on actual converted full windows, on CPU."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    actor = TinyFastWAM()
    _tiny_text_cache(dataset, actor)
    data_cfg = OmegaConf.load(dataset / "data_idm.yaml")
    train = instantiate(data_cfg.train)
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                train, batch_size=min(8, len(train)), shuffle=True, num_workers=0
            )
        )
    )
    adapter, report, parent_state = adapt_tiny_fixture(actor, batch=batch, steps=steps)
    parent = output / "W0_tiny.pt"
    torch.save({"kind": "tiny_cpu_fixture", "state_dict": parent_state}, parent)
    adapter.save_sidecar(
        output / "U_BC_tiny.pt",
        parent_checkpoint_sha256=sha256_file(parent),
        extra_metadata={"kind": "tiny_cpu_fixture", "dataset": str(dataset)},
    )
    processor_path = output / "processor.yaml"
    OmegaConf.save(
        OmegaConf.create({"processor": data_cfg.train.processor}), processor_path
    )
    report.update(
        {
            "status": "PASS",
            "scope": "CPU/tiny on converted mock demonstration windows",
            "real_assets": "REAL-ASSET-NOT-RUN",
            "dataset": str(dataset),
            "parent": str(parent),
            "sidecar": str(output / "U_BC_tiny.pt"),
            "stats": str(dataset / "dataset_stats.json"),
            "processor_config": str(processor_path),
            "text_cache": str(dataset / "text_cache"),
        }
    )
    (output / "initialization.json").write_text(json.dumps(report, indent=2))
    return report


def run_uncond_bc(config):
    """Run explicitly configured real BC with the production current-only policy."""
    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError("Missing real BC settings: " + ", ".join(sorted(missing)))
    dataset, output = Path(config.dataset), Path(config.output_dir)
    if config.kind != "fastwam":
        raise ValueError("Use --tiny for the separate CPU fixture.")
    if config.max_steps < 1 or config.micro_batch_size < 1 or config.learning_rate <= 0:
        raise ValueError(
            "Offline BC requires positive steps, batch size and learning rate."
        )
    torch.manual_seed(config.seed)
    output.mkdir(parents=True, exist_ok=False)
    model_cfg = OmegaConf.load(config.model_config)
    actor = instantiate(
        model_cfg.fastwam,
        model_dtype=getattr(torch, config.precision),
        device=config.device,
    )
    load_strict_fastwam_parent(actor, str(config.parent_checkpoint))
    data_cfg = OmegaConf.load(dataset / "data_uncond_bc.yaml")
    train = instantiate(data_cfg.train)
    example = train[0]
    policy = FastWAMUncondBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(rank=16, alpha=16),
        config=FastWAMUncondBCConfig(
            expected_video_frames=1,
            expected_video_height=example["video"].shape[-2],
            expected_video_width=example["video"].shape[-1],
        ),
    )
    optimizer = torch.optim.AdamW(
        policy.lora_adapter.lora_parameters(),
        lr=config.learning_rate,
        betas=tuple(config.betas),
        weight_decay=config.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        train, batch_size=config.micro_batch_size, shuffle=True, num_workers=0
    )
    losses = []
    while len(losses) < config.max_steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = policy(batch)["loss_action_bc"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Offline BC produced non-finite loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.lora_adapter.lora_parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) == config.max_steps:
                break
    policy.lora_adapter.save_sidecar(
        output / "U_BC_real.pt",
        parent_checkpoint_sha256=sha256_file(config.parent_checkpoint),
        extra_metadata={
            "dataset": str(dataset),
            "stats": str(dataset / "dataset_stats.json"),
        },
    )
    report = {
        "steps": len(losses),
        "losses": losses,
        "parent": str(config.parent_checkpoint),
        "sidecar": str(output / "U_BC_real.pt"),
    }
    (output / "initialization.json").write_text(json.dumps(report, indent=2))
    OmegaConf.save(config, output / "config.yaml")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Offline real-data adaptation (no robot APIs)"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.tiny:
        if args.dataset is None or args.output is None:
            parser.error("--tiny requires --dataset and --output")
        result = run_tiny_adaptation(args.dataset, args.output, steps=args.steps)
    elif args.config:
        result = run_uncond_bc(OmegaConf.load(args.config))
    else:
        parser.error("Specify --tiny or an explicit real --config")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
