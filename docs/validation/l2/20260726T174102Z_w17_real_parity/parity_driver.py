#!/usr/bin/env python
"""Phase B2: W17 real-hardware bitwise parity driver (TEST_PLAN section 3.3).

One script, used byte-identically in TWO FastWAM checkouts:

  * main checkout  ff4d056  (post-W17, metric-adaptive HEAD)
  * baseline worktree 7428d72 (pre-W17 parity baseline)

It never imports anything that exists in only one checkout (no
fastwam.diagnostics.verify_taps — absent at 7428d72). Model construction
replicates the W22 harness pattern (hydra compose -> instantiate ->
load_checkpoint -> eval), and the synthetic sample replicates
build_synthetic_sample: all zeros/ones, no RNG, so the sample itself is
bitwise identical across processes.

Modes
-----
seeds       (both checkouts)  For each seed in {0,1,2} run
              (a) the PUBLIC ``model.infer_action`` — on this task's model
                  class (FastWAMIDM) that is the forced-IDM pipeline
                  (infer_joint: full video denoise + teacher-forced action
                  denoise). Cross-checkout bitwise equality of this output is
                  merge-precondition-2 (forced-IDM parity).
              (b) the base/UNCOND solver via EXPLICIT class dispatch
                  ``FastWAM.infer_action(model, ...)`` — the exact dispatch
                  pattern MetricAdaptiveFastWAM._call_inherited_branch uses
                  for branch "base". This is the code path W17 actually
                  refactored (_infer_action_impl), so its cross-checkout
                  parity verifies the refactor is bitwise-neutral.
              Saves every final action tensor (CPU, exact returned dtype).
gradparity  (main checkout only)  Seed 0: FastWAM.infer_action vs
              FastWAM.infer_action_with_grad under otherwise DEFAULT kwargs.
              The grad result is normalized with the *identical* conversion
              infer_action applies (`.detach().to(cpu, float32)`) before
              comparison. Criterion: torch.equal AND max_abs == 0.
mem         (main checkout only)  Peak CUDA memory for
              FastWAM.infer_action_with_grad at num_inference_steps in
              {5,10,20}, seed 0. Non-gating (WS5 budget input).

No FastWAM source file is imported-and-monkeypatched, edited, or shadowed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

CKPT = (
    "/root/autodl-fs/fastwam/runs/libero_idm_2cam224_1e-4/"
    "torch_compile_speedup_full_wanrobot_4gpu_bs8_ga4_20260726_141000/"
    "checkpoints/weights/step_002000.pt"
)
TASK = "libero_idm_2cam224_1e-4"
OUT_DIR = Path("/root/autodl-tmp/.tmp/phaseB2/tensors")
SEEDS = (0, 1, 2)
NUM_INFERENCE_STEPS = 20
ACTION_HORIZON = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def filter_kwargs(func, kwargs):
    """Signature-driven kwarg filter (same semantics as the W22 harness's
    _call_with_supported_kwargs, replicated here because that module does not
    exist in the 7428d72 baseline)."""
    parameters = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in parameters}


def reseed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_report() -> tuple[dict, Path]:
    import fastwam  # resolved via PYTHONPATH -> proves which checkout runs

    # src layout: <repo_root>/src/fastwam/__init__.py
    repo_root = Path(fastwam.__file__).resolve().parents[2]
    git_sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    info = {
        "repo_root": str(repo_root),
        "cwd": os.getcwd(),
        "fastwam_file": str(Path(fastwam.__file__).resolve()),
        "git_head": git_sha,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for key, value in sorted(info.items()):
        log(f"env {key} = {value}")
    return info, repo_root


def build_model_and_sample(repo_root: Path):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_dir = repo_root / "configs"
    if not (config_dir / "train.yaml").exists():
        raise SystemExit(f"no configs/train.yaml under {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])

    log("instantiating model (bf16, cuda) ...")
    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(CKPT)
    model = model.to("cuda").eval()
    log(f"model ready in {time.time() - t0:.1f}s: {type(model).__module__}.{type(model).__name__}")
    log(f"model mro: {[c.__name__ for c in type(model).__mro__]}")
    n_req = sum(1 for p in model.parameters() if p.requires_grad)
    n_all = sum(1 for p in model.parameters())
    log(f"params requiring grad: {n_req}/{n_all}")

    video_size = OmegaConf.select(cfg, "data.train.video_size", default=None)
    image_hw = tuple(int(v) for v in video_size) if video_size is not None else (224, 448)
    context_len = int(OmegaConf.select(cfg, "model.tokenizer_max_len", default=128))
    context_dim = int(
        OmegaConf.select(cfg, "model.action_dit_config.text_dim", default=None)
        or OmegaConf.select(cfg, "model.video_dit_config.text_dim", default=None)
        or 4096
    )
    num_video_frames = int(OmegaConf.select(cfg, "data.train.num_frames", default=33))

    sample = {
        "prompt": None,
        "input_image": torch.zeros(1, 3, image_hw[0], image_hw[1]),
        "action_horizon": ACTION_HORIZON,
        "context": torch.zeros(1, context_len, context_dim),
        "context_mask": torch.ones(1, context_len, dtype=torch.bool),
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_video_frames": num_video_frames,
        "rand_device": "cpu",
    }
    proprio_dim = getattr(model, "proprio_dim", None)
    if proprio_dim is not None:
        sample["proprio"] = torch.zeros(1, int(proprio_dim))
    log(
        f"sample: image_hw={image_hw} context=({context_len},{context_dim}) "
        f"num_video_frames={num_video_frames} proprio_dim={proprio_dim} "
        f"steps={NUM_INFERENCE_STEPS} horizon={ACTION_HORIZON}"
    )
    return model, sample


def save_action(tag: str, action: torch.Tensor, meta: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = action.detach().cpu()
    path = OUT_DIR / f"{tag}.pt"
    torch.save(t, str(path))
    record = {
        "tag": tag,
        "path": str(path),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": tensor_sha256(t),
        **meta,
    }
    log(f"saved {tag}: shape={record['shape']} dtype={record['dtype']} sha256={record['sha256']}")
    return record


def run_seeds(model, sample, label: str) -> list[dict]:
    from fastwam.models.wan22.fastwam import FastWAM

    records = []
    for seed in SEEDS:
        # (a) PUBLIC infer_action: on FastWAMIDM this is the forced-IDM
        #     pipeline (merge-precondition-2 cross-checkout criterion).
        reseed(seed)
        kwargs = filter_kwargs(model.infer_action, {**sample, "seed": seed})
        t0 = time.time()
        out = model.infer_action(**kwargs)
        torch.cuda.synchronize()
        wall = time.time() - t0
        records.append(
            save_action(
                f"{label}_idm_seed{seed}",
                out["action"],
                {"entry": f"{type(model).__name__}.infer_action(public)", "seed": seed,
                 "wall_s": round(wall, 2), "kwargs_used": sorted(kwargs.keys())},
            )
        )
        del out

        # (b) base/UNCOND solver via explicit class dispatch — the W17
        #     refactored path (_infer_action_impl). Same dispatch pattern as
        #     MetricAdaptiveFastWAM._call_inherited_branch branch "base".
        reseed(seed)
        base_kwargs = filter_kwargs(FastWAM.infer_action, {**sample, "seed": seed})
        t0 = time.time()
        out = FastWAM.infer_action(model, **base_kwargs)
        torch.cuda.synchronize()
        wall = time.time() - t0
        records.append(
            save_action(
                f"{label}_base_seed{seed}",
                out["action"],
                {"entry": "FastWAM.infer_action(explicit dispatch)", "seed": seed,
                 "wall_s": round(wall, 2), "kwargs_used": sorted(base_kwargs.keys())},
            )
        )
        del out
        gc.collect()
        torch.cuda.empty_cache()
    return records


def run_gradparity(model, sample, label: str) -> list[dict]:
    from fastwam.models.wan22.fastwam import FastWAM

    seed = 0
    base_kwargs = filter_kwargs(FastWAM.infer_action, {**sample, "seed": seed})

    reseed(seed)
    t0 = time.time()
    out_nograd = FastWAM.infer_action(model, **base_kwargs)
    torch.cuda.synchronize()
    wall_nograd = time.time() - t0
    a = out_nograd["action"].detach().cpu()

    reseed(seed)
    grad_kwargs = filter_kwargs(FastWAM.infer_action_with_grad, {**sample, "seed": seed})
    t0 = time.time()
    out_grad = FastWAM.infer_action_with_grad(model, **grad_kwargs)
    torch.cuda.synchronize()
    wall_grad = time.time() - t0
    graph_carrying = out_grad["action"].requires_grad and out_grad["action"].grad_fn is not None
    # Normalize with the IDENTICAL conversion infer_action applies to the same
    # latent: .detach().to(device="cpu", dtype=torch.float32).
    b = out_grad["action"].detach().to(device="cpu", dtype=torch.float32)
    del out_grad
    gc.collect()
    torch.cuda.empty_cache()

    equal = torch.equal(a, b)
    max_abs = float((a.double() - b.double()).abs().max().item()) if a.shape == b.shape else float("nan")
    records = [
        save_action(f"{label}_gradparity_nograd_seed{seed}", a,
                    {"entry": "FastWAM.infer_action(explicit dispatch)", "seed": seed,
                     "wall_s": round(wall_nograd, 2)}),
        save_action(f"{label}_gradparity_grad_seed{seed}", b,
                    {"entry": "FastWAM.infer_action_with_grad(explicit dispatch, normalized cpu/fp32)",
                     "seed": seed, "wall_s": round(wall_grad, 2),
                     "graph_carrying_before_normalize": bool(graph_carrying)}),
    ]
    verdict = {
        "check": "gradparity",
        "seed": seed,
        "torch_equal": bool(equal),
        "max_abs": max_abs,
        "graph_carrying": bool(graph_carrying),
        "wall_nograd_s": round(wall_nograd, 2),
        "wall_grad_s": round(wall_grad, 2),
    }
    log(f"GRADPARITY VERDICT: {json.dumps(verdict, sort_keys=True)}")
    records.append(verdict)
    return records


def run_mem(model, sample, label: str) -> list[dict]:
    from fastwam.models.wan22.fastwam import FastWAM

    seed = 0
    records = []
    model_footprint = int(torch.cuda.memory_allocated())
    log(f"model footprint before mem runs: {model_footprint / 2**20:.1f} MiB")
    for steps in (5, 10, 20):
        reseed(seed)
        kwargs = filter_kwargs(
            FastWAM.infer_action_with_grad,
            {**sample, "seed": seed, "num_inference_steps": steps},
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = FastWAM.infer_action_with_grad(model, **kwargs)
        torch.cuda.synchronize()
        wall = time.time() - t0
        peak = int(torch.cuda.max_memory_allocated())
        rec = {
            "check": "memory",
            "entry": "FastWAM.infer_action_with_grad",
            "num_inference_steps": steps,
            "seed": seed,
            "peak_cuda_MiB": round(peak / 2**20, 1),
            "model_footprint_MiB": round(model_footprint / 2**20, 1),
            "wall_s": round(wall, 2),
        }
        log(f"MEMORY: {json.dumps(rec, sort_keys=True)}")
        records.append(rec)
        del out
        gc.collect()
        torch.cuda.empty_cache()
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("seeds", "gradparity", "mem"), required=True)
    parser.add_argument("--label", required=True, help="checkout label: main | base")
    args = parser.parse_args()

    info, repo_root = env_report()
    log(f"checkpoint: {CKPT}")
    ckpt_sha = sha256_file(CKPT)
    log(f"checkpoint sha256: {ckpt_sha}")
    info["ckpt_sha256"] = ckpt_sha

    model, sample = build_model_and_sample(repo_root)

    if args.mode == "seeds":
        records = run_seeds(model, sample, args.label)
    elif args.mode == "gradparity":
        records = run_gradparity(model, sample, args.label)
    else:
        records = run_mem(model, sample, args.label)

    result = {"mode": args.mode, "label": args.label, "env": info, "records": records}
    out_path = Path("/root/autodl-tmp/.tmp/phaseB2") / f"{args.label}_{args.mode}_result.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
