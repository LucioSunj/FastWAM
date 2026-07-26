# W17 real-hardware bitwise parity verification (Phase B2, TEST_PLAN section 3.3)

Verdict: **PASS** on both gating criteria, with zero tolerance applied.

1. **Cross-checkout forced-output parity (merge-precondition-2)**: for every
   seed in {0, 1, 2} and both inference paths, the final action tensor
   produced at `ff4d056` (post-W17 `metric-adaptive` HEAD) is bitwise
   identical to the one produced at `7428d72` (pre-W17 baseline):
   `torch.equal == True` and `max_abs == 0.0` on all 6 pairs.
2. **W17 no-grad-default grad parity**: on `ff4d056`, seed 0,
   `FastWAM.infer_action` vs `FastWAM.infer_action_with_grad` under otherwise
   default kwargs: `torch.equal == True`, `max_abs == 0.0`, and the grad
   result was verified graph-carrying (`requires_grad=True`, `grad_fn` set)
   before normalization.

Non-gating: peak CUDA memory of `infer_action_with_grad` at
`num_inference_steps` in {5, 10, 20} is recorded below (WS5 budget input).

## Setting

- Date (UTC): 2026-07-26. Runs 17:41:02Z – 17:51:54Z.
- Host: `autodl-a800-49645`, 1x NVIDIA A800 80GB PCIe, otherwise idle; one
  model job at a time.
- Python 3.10.20 (`/root/autodl-tmp/fastwam-env/bin/python`),
  torch 2.7.1+cu128, CUDA 12.8, cuDNN 90701.
- Numerics flags identical in every run (library defaults, recorded in each
  log): `cudnn_benchmark=False`, `cudnn_deterministic=False`,
  `matmul_allow_tf32=False`, `cudnn_allow_tf32=True`,
  `float32_matmul_precision=highest`. Model dtype BF16, device cuda.
- Environment exports for every run:

```
PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
HF_HOME=/root/autodl-tmp/.cache/huggingface
TMPDIR=/root/autodl-tmp/.tmp
WORKSPACE_ROOT=/root/When-will-inference-time-prediction-beneficial-
FASTWAM_WAN22_COMPONENT_DIR=/root/autodl-fs/fastwam/models/wan2.2-ti2v-5b-components
PATH=/root/autodl-tmp/fastwam-env/bin:$PATH
PYTHONPATH="src:."   # relative to the respective checkout root
```

## Checkouts under comparison (SHAs verbatim from `git rev-parse HEAD`)

| role | path | HEAD |
|---|---|---|
| main (post-W17) | `/root/When-will-inference-time-prediction-beneficial-/FastWAM` | `ff4d05608d85068db340215c35356084ff702f64` |
| baseline (pre-W17) | `/root/autodl-tmp/.tmp/parity_base` (git worktree) | `7428d722ad389b437f93582578d994d4b2984f59` |

Each run's log records `fastwam.__file__` and the checkout's `git rev-parse
HEAD` from inside the process, proving which code executed. Source scope of
the change under test (full-tree `diff -rq` between the checkouts, ignoring
`__pycache__`): only `src/fastwam/models/wan22/fastwam.py`,
`src/fastwam/models/wan22/fastwam_metric_adaptive.py`,
`src/fastwam/adaptive_gate/wam_mode_adapter.py` differ; `src/fastwam/adapters/`
and `src/fastwam/diagnostics/verify_taps.py` exist only in main. `configs/`
(task/model/data), `src/fastwam/runtime.py`, `fastwam_idm.py`,
`fastwam_joint.py`, and the Wan loader are byte-identical across the two
checkouts.

## Model / checkpoint lineage

- Task config: `libero_idm_2cam224_1e-4` (hydra-composed from the respective
  checkout's `configs/`; model `fastwam_idm` -> `FastWAMIDM`, MRO
  `FastWAMIDM -> FastWAMJoint -> FastWAM`).
- Checkpoint (12,041,735,573 bytes, step 2000, WanRobot-backbone variant run):
  `/root/autodl-fs/fastwam/runs/libero_idm_2cam224_1e-4/torch_compile_speedup_full_wanrobot_4gpu_bs8_ga4_20260726_141000/checkpoints/weights/step_002000.pt`
  - SHA256 (recomputed inside every run): `b13e70a3cc9a7f63a47dca13f5f65eb294baf0d2aaa7b384854d6591a42474d1`
- Loaded via the public `model.load_checkpoint` (identical code in both
  checkouts); benign known warning: checkpoint carries no FastWAM provenance
  metadata.
- Wan2.2 components from `FASTWAM_WAN22_COMPONENT_DIR` (same directory for
  both runs); ActionDiT backbone resolves through
  `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt ->
  /root/autodl-fs/fastwam/checkpoints/...` (same physical file in both
  checkouts, see Deviations).

## Fixed synthetic sample (bitwise-identical across processes; no RNG)

Replicates `fastwam.diagnostics.verify_taps.build_synthetic_sample` (W22/B1
harness) with dims read from the composed config: `input_image` zeros
`[1,3,224,448]`, `context` zeros `[1,128,4096]`, `context_mask` ones,
`proprio` zeros `[1,8]`, `prompt=None`, `action_horizon=32`,
`num_video_frames=33`, `num_inference_steps=20`, `rand_device="cpu"`, BF16.
Per call: `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`, and
the same `seed` passed to the inference entry point. One process per checkout
runs all calls in the same order (model loaded once).

## Inference paths exercised (why there are two)

- **`idm_*`: the public `model.infer_action`.** On this task's model class
  that is `FastWAMIDM.infer_action`, i.e. the forced-IDM pipeline
  (`FastWAMIDM.infer_joint`: full 20-step video denoise + teacher-forced
  action denoise). Its cross-checkout bitwise equality is exactly
  merge-precondition-2 ("forced-IDM output max_abs == 0 against the
  pre-change checkpoint").
- **`base_*`: the base/UNCOND solver via explicit class dispatch
  `FastWAM.infer_action(model, ...)`.** `FastWAMIDM` overrides
  `infer_action`, so the public call never enters the code W17 actually
  refactored (`FastWAM._infer_action_impl` /
  `_predict_action_noise_with_cache_impl`). The explicit dispatch is the same
  pattern `MetricAdaptiveFastWAM._call_inherited_branch` uses for branch
  `"base"`, and it drives the refactored solver directly. Without this leg
  the cross-checkout comparison would not touch the refactored lines at all.

Both legs use only public methods; no FastWAM source file was edited.

## Results: cross-checkout bitwise parity (gating; criterion verbatim: torch.equal AND max_abs == 0, NO tolerance)

All tensors `[32, 7]` float32 (the exact object `infer_action` returns,
saved on CPU). SHA256 is over the tensor's raw bytes.

| pair | seed | torch.equal | max_abs | SHA256 (identical main/base) |
|---|---|---|---|---|
| public forced-IDM | 0 | True | 0.0 | `80764faae0e4fd06627ce1a6a6fdb56134cdbf040992ef1348265f3e919e7c6f` |
| public forced-IDM | 1 | True | 0.0 | `a410accfebd603afcdde8ac78e8438c2802570a2fc6907a3a34e62541325c096` |
| public forced-IDM | 2 | True | 0.0 | `b09162a7fae1d765b02067075d7574f80fab944a24f43e1c515bb65da971a6dd` |
| explicit-dispatch UNCOND solver | 0 | True | 0.0 | `cc3bba19d80aac9cf87f347d2d5f1cca9c28bba00e483d5afb718d8830fedb9a` |
| explicit-dispatch UNCOND solver | 1 | True | 0.0 | `2ede4bc4048a1773977db6d029b7d813401e5a9c48d90d95349afbd602fedfae` |
| explicit-dispatch UNCOND solver | 2 | True | 0.0 | `6b45ae4b312478fbb25712e4a0909e3bbe5b57c184d63c3f1aeded52149e0a0a` |

`parity_compare.py` rc = 0 (`compare_result.json`:
`all_pairs_bitwise_equal: true`).

Wall times (main / base, seconds): IDM 2.63/2.71, 2.33/2.36, 2.32/2.30;
UNCOND 0.96/0.92, 0.96/0.94, 0.94/0.93 (seeds 0/1/2). Model load
65.8s / 67.1s; run rc = 0 / 0.

## Results: same-checkout grad parity (gating; main checkout only, seed 0)

`FastWAM.infer_action` vs `FastWAM.infer_action_with_grad`, otherwise
identical default kwargs and the same fixed sample. Per the W17 contract the
grad entry returns the on-device BF16 graph-carrying latent; it was verified
graph-carrying (`requires_grad=True` and `grad_fn is not None` before
normalization) and then normalized with the *identical* conversion
`infer_action` applies to the same latent
(`.detach().to(device="cpu", dtype=torch.float32)`) before comparison.

| check | torch.equal | max_abs | graph_carrying | wall no-grad | wall grad |
|---|---|---|---|---|---|
| gradparity seed 0, 20 steps | True | 0.0 | True | 1.27 s | 1.68 s |

Both tensors' SHA256:
`cc3bba19d80aac9cf87f347d2d5f1cca9c28bba00e483d5afb718d8830fedb9a` — which
also equals the seeds-run `base_seed0` digest from BOTH checkouts, i.e. the
UNCOND result is additionally reproduced bitwise across separate processes.
Run rc = 0.

## Results: infer_action_with_grad peak CUDA memory (non-gating; WS5 budget input)

`FastWAM.infer_action_with_grad`, seed 0, one run per point, BF16, A800 80GB.
`torch.cuda.reset_peak_memory_stats()` before and
`torch.cuda.max_memory_allocated()` after each run;
1651/1847 parameters require grad (construction default — nothing frozen by
the driver), so the whole solver loop is recorded, as WS5 training would.

| num_inference_steps | peak CUDA (MiB) | model footprint (MiB) | graph overhead vs footprint (MiB) | wall (s) |
|---|---|---|---|---|
| 5 | 15,082.0 | 12,843.0 | 2,239.0 | 0.74 |
| 10 | 16,426.3 | 12,843.0 | 3,583.3 | 0.82 |
| 20 | 19,115.5 | 12,843.0 | 6,272.5 | 1.55 |

Scaling is linear at ~269 MiB per additional recorded solver step for this
shape (1x [224,448] two-camera frame, horizon 32, BF16), confirming the
documented W17 property that memory grows with `num_inference_steps`.
Run rc = 0.

## Deviations / notes

1. Baseline worktree placed at `/root/autodl-tmp/.tmp/parity_base` instead of
   the plan's literal `/tmp/parity_base` (disk space on the data volume), as
   pre-authorized by the phase instructions.
2. The first baseline launch failed at model construction:
   `FileNotFoundError: .../parity_base/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`.
   The model config references `./checkpoints/...` relative to the checkout,
   and that directory is a gitignored asset dir absent from a fresh worktree.
   Fix: `ln -s /root/When-will-inference-time-prediction-beneficial-/FastWAM/checkpoints
   /root/autodl-tmp/.tmp/parity_base/checkpoints` (untracked symlink in a
   gitignored path; the main checkout's entry is itself a symlink to
   `/root/autodl-fs/fastwam/checkpoints/...`, so both runs read the same
   physical file). This is an environmental fix made before any baseline
   parity number existed — not a rerun-to-pass.
3. `fastwam.diagnostics.verify_taps` is not importable at `7428d72` (predates
   W22), so the driver replicates the harness inline from public APIs present
   in both checkouts: hydra compose of `configs/train.yaml` with
   `task=libero_idm_2cam224_1e-4`, `instantiate(cfg.model,
   model_dtype=bf16, device=cuda)`, `model.load_checkpoint`, plus verbatim
   re-implementations of `build_synthetic_sample` and the signature-driven
   kwarg filter.
4. On this model class the public `infer_action` is the IDM pipeline; the
   W17-refactored UNCOND solver is additionally exercised via explicit
   `FastWAM.infer_action` dispatch (see "Inference paths exercised").
   Per-step solver intermediates were not captured: the W8
   `velocity_hook`/`return_init_noise` knobs exist in BOTH checkouts only on
   the base/UNCOND path, and defaults were kept deliberately ("default
   arguments" criterion); the cross-checkout criterion is the final action
   tensor.
5. The IDM leg passes `num_video_frames=33` (required, no default on
   `FastWAMIDM.infer_action`); the signature filter drops it for
   `FastWAM.infer_action`, which does not accept it. All other kwargs are
   entry-point defaults except the fixed sample and `seed`.
6. After all runs, `git worktree remove /root/autodl-tmp/.tmp/parity_base`
   succeeded cleanly — no `--force` needed (the added `checkpoints` symlink
   and `__pycache__` live under gitignored paths).

## Artifacts in this directory

- `parity_driver.py`, `parity_compare.py` — the scripts (used byte-identically
  in both checkouts).
- `main_seeds.log` / `.rc`, `base_seeds.log` / `.rc`, `main_gradparity.log` /
  `.rc`, `main_mem.log` / `.rc`, `compare.log` — full run logs and exit codes.
- `main_seeds_result.json`, `base_seeds_result.json`, `compare_result.json`,
  `main_gradparity_result.json`, `main_mem_result.json` — structured results.
- `tensors/` — every saved action tensor (`*.pt`, CPU float32, 1.7 KiB each).
- `SHA256SUMS.txt` — digest of every file above.

Raw tensors also remain at `/root/autodl-tmp/.tmp/phaseB2/tensors/`.
