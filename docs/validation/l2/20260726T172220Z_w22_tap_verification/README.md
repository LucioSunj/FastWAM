# W22 tap verification on real hardware (Phase B1, TEST_PLAN section 3.1)

Evidence for the L2 ladder step: run `fastwam.diagnostics.verify_taps` against
a real FastWAM model + checkpoint on GPU, and — for the pre-registered
`vae_latent` zero-fire outcome — execute the documented fallback (first-frame
VAE latent obtained via the `encode_world_state` path, fed to
`fastwam.diagnostics.probe`).

Verdict: **GREEN** (criteria below). The `vae_latent` tap fired 0 times,
exactly as pre-registered; both hookable taps fired and the fallback ran clean
end-to-end.

## Setting

- Date (UTC): 2026-07-26. Real harness run started 20260726T172220Z.
- Host: `autodl-a800-49645`, 1x NVIDIA A800 80GB PCIe, otherwise idle.
- Repo: `/root/When-will-inference-time-prediction-beneficial-`
  (outer `master` @ `2041381`; `FastWAM` @ `ff4d056` on `metric-adaptive`, clean).
- Python 3.10.20 (`/root/autodl-tmp/fastwam-env/bin/python`),
  torch 2.7.1+cu128, transformers 4.49.0.
- Environment exports for every run:

```
PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
HF_HOME=/root/autodl-tmp/.cache/huggingface
TMPDIR=/root/autodl-tmp/.tmp
WORKSPACE_ROOT=/root/When-will-inference-time-prediction-beneficial-
FASTWAM_WAN22_COMPONENT_DIR=/root/autodl-fs/fastwam/models/wan2.2-ti2v-5b-components
PATH=/root/autodl-tmp/fastwam-env/bin:$PATH
```

## Model / checkpoint lineage

- Task config: `libero_idm_2cam224_1e-4` (composed from the FastWAM repo
  `configs/`; model `fastwam_idm` -> `FastWAMIDM`).
- Checkpoint (12,041,735,573 bytes, trained 2026-07-26, WanRobot-backbone
  variant run, step 2000):
  `/root/autodl-fs/fastwam/runs/libero_idm_2cam224_1e-4/torch_compile_speedup_full_wanrobot_4gpu_bs8_ga4_20260726_141000/checkpoints/weights/step_002000.pt`
  - SHA256: `b13e70a3cc9a7f63a47dca13f5f65eb294baf0d2aaa7b384854d6591a42474d1`
- Dataset stats (same run dir, `dataset_stats.json`):
  - SHA256: `8f54b82fc72b363fc486d6910cc49e72d440b6f2d0004b7680787851ce8e1ecb`
- Model config digest (as computed by `_construct_real_model`, recorded by the
  fallback run): `db612608a3607759123fea342df4cf99b0fcc0977ef5c0883b19f45b2729ca16`
- Benign warning emitted by `load_checkpoint` in both runs: the checkpoint
  carries no FastWAM provenance metadata (expected for this training variant;
  verify_taps does not require provenance).

## Commands (verbatim, run from the FastWAM repo root)

1. Self-test (CPU, rc=0 — `verify_taps_selftest.log`):

```
PYTHONPATH="src:." python -m fastwam.diagnostics.verify_taps --self-test
```

2. Real run (rc=3 — `verify_taps_run.log`; rc captured from the nohup rc-file):

```
PYTHONPATH="src:." python -m fastwam.diagnostics.verify_taps \
  --task libero_idm_2cam224_1e-4 \
  --ckpt /root/autodl-fs/fastwam/runs/libero_idm_2cam224_1e-4/torch_compile_speedup_full_wanrobot_4gpu_bs8_ga4_20260726_141000/checkpoints/weights/step_002000.pt \
  --dataset-stats /root/autodl-fs/fastwam/runs/libero_idm_2cam224_1e-4/torch_compile_speedup_full_wanrobot_4gpu_bs8_ga4_20260726_141000/dataset_stats.json \
  --out runs/diagnostics/tap_verification.json \
  --device cuda
```

Wall time 87 s (epoch 1785086540 -> 1785086627). Peak GPU memory 15,529 MiB
(`nvidia-smi` sampled every 5 s, 18 samples — `verify_taps_gpumem_5s.log`).
Defaults in effect: `--dtype bf16`, `--steps 2`, `--action-horizon 32`,
`--seed 0`, `--pool-output-dim 64`.

3. Fallback check (rc=0 — `vae_fallback_check.log`, JSON in
`vae_fallback_check.json`; script copied here as `vae_fallback_check.py`,
canonical location `FastWAM/runs/diagnostics/vae_fallback_check.py`):

```
PYTHONPATH="src:." python runs/diagnostics/vae_fallback_check.py \
  --task libero_idm_2cam224_1e-4 \
  --ckpt /root/autodl-fs/fastwam/runs/libero_idm_2cam224_1e-4/torch_compile_speedup_full_wanrobot_4gpu_bs8_ga4_20260726_141000/checkpoints/weights/step_002000.pt \
  --device cuda \
  --out runs/diagnostics/vae_fallback_check.json
```

Wall time 66 s. Peak GPU memory 15,867 MiB (`vae_fallback_gpumem_5s.log`).

## Per-tap results (single forward, `num_inference_steps=2`)

| tap | module_path | site | fired | shape | dtype | device | feature_dim | pooled_feature_dim |
|---|---|---|---|---|---|---|---|---|
| `vae_latent` | `vae` | output | **0** | — | — | — | 1 | — |
| `video_block_0` | `video_expert.blocks.0` | output | 2 | [1, 882, 3072] | torch.bfloat16 | cuda:0 | -1 | 64 |
| `action_readout` | `action_expert.head` | input | 2 | [1, 32, 1024] | torch.bfloat16 | cuda:0 | -1 | 64 |

`fired=2` matches the two solver steps re-entering the modules; the harness
reports (never hides) multi-fire counts. `shape_varies=false` for both.

The `vae_latent` zero-fire is the **pre-registered expected outcome** (W22
finding, documented in both `fastwam/diagnostics/taps.py` and
`fastwam/diagnostics/verify_taps.py` module docstrings):
`FastWAM._encode_input_image_latents_tensor` reaches the VAE via the plain
method chain `self.vae.encode(...)` (`WanVideoVAE38.encode`,
`wan_video_vae.py:1218`), which never enters `Module.__call__`, so a forward
hook on the `vae` submodule cannot fire. The harness fail-closed with rc=3 and
the exact error text is in `verify_taps_run.log`.

## Fallback outcome (documented route for locus (a))

- The documented adapter entry point was attempted first and fail-closed on
  this plain IDM checkpoint, recorded verbatim:
  `TypeError: WAMModeAdapter requires a current dual-regime IDM model whose
  `infer_action` accepts ['first_frame_latents', 'force_branch'];
  missing=['force_branch'].`
- The script therefore performed the exact call `encode_world_state` makes
  internally (`model._encode_input_image_latents_tensor`,
  `adaptive_gate/wam_mode_adapter.py:325-348`), reproducing the
  `EncodedWorldState` contract checks verbatim:
  - first-frame latent: shape `[1, 48, 1, 14, 28]`, `torch.bfloat16`,
    `cuda:0`, all-finite; `vae.model.z_dim = 48`.
  - world-feature contract: 240 == 5 * z_dim = 240 (PASS).
- End-to-end probe acceptance: 20 deterministic seeded synthetic inputs
  (sample 0 = the fixed input) encoded and stacked to `[20, 48, 1, 14, 28]`;
  `pool_activation(feature_dim=1)` -> `[20, 64]`; `probe_taps` (grouped
  5-fold cross-fitted linear probe) ran clean. The AUC (0.65) is on synthetic
  alternating labels and is meaningless by construction — the acceptance
  criterion is shapes/dtypes accepted end-to-end without error, which held.

## Acceptance evaluation (TEST_PLAN section 3.1 criteria, machine outputs only)

GREEN iff: `video_block` fired>=1 AND `action_readout` fired>=1 with
shapes/feature_dim recorded, AND (`vae_latent` fired>=1 OR fallback check runs
clean end-to-end).

- `video_block_0` fired = 2 >= 1, shape/feature_dim recorded: PASS
- `action_readout` fired = 2 >= 1, shape/feature_dim recorded: PASS
- `vae_latent` fired = 0; fallback rc=0, clean end-to-end: PASS (via fallback)

**Verdict: GREEN.**

## Deviations / notes

1. `runs/diagnostics/tap_verification.json` was **not** written: the harness
   fail-closes on any zero-fire tap (rc=3) and deliberately writes no `--out`
   file "that could be mistaken for a pass" (`verify_taps.py`, `main()`).
   The per-tap data for the taps that did fire was printed to stderr and is
   preserved verbatim in `verify_taps_run.log` and extracted (unmodified) in
   `taps_fired.json`.
2. The fallback used `model._encode_input_image_latents_tensor` (the identical
   underlying call) because `WAMModeAdapter` fail-closes on a non-dual-regime
   model; the constructor error is recorded verbatim in
   `vae_fallback_check.json` under `adapter_error`. This matches the fallback
   as documented in `taps.py`: locus (a) is obtained by keeping the returned
   first-frame latent and passing it straight to `probe_taps`.
3. rc values were captured from nohup rc-files, not inferred from log tails:
   self-test 0, real run 3, fallback 0.
4. No model source, no existing test, and no existing validation dir was
   modified. New files: this directory and
   `FastWAM/runs/diagnostics/vae_fallback_check.{py,json}` (runs/ is
   untracked).

## Files

- `verify_taps_selftest.log` — self-test stdout (rc=0)
- `verify_taps_run.log` — real run stdout+stderr (rc=3, fail-closed)
- `taps_fired.json` — per-tap firings extracted verbatim from the stderr JSON
- `vae_fallback_check.py` — fallback script (copy)
- `vae_fallback_check.json` — fallback machine output (rc=0)
- `vae_fallback_check.log` — fallback stdout+stderr
- `verify_taps_gpumem_5s.log`, `vae_fallback_gpumem_5s.log` — 5 s `nvidia-smi`
  memory.used samples (MiB) during each run
- `SHA256SUMS.txt` — SHA256 over every file above (excludes itself)
