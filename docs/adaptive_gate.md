# Binary adaptive world-model reasoning

The external RL gate makes exactly one categorical decision per generated action
chunk. The public mode order is stable:

| index | mode | frozen WAM path |
|---:|---|---|
| 0 | `UNCOND` | `force_branch="base"`: current-frame action inference, no future denoising |
| 1 | `IDM` | `force_branch="idm"`: complete future-latent generation followed by the full action solver |

There is no low-step or joint choice. A low-NFE solve is a different numerical
solver, not an intermediate state of the complete trajectory, and would confound
world reasoning with solver quality.

## Required checkpoint

Train the fused dual-regime IDM model so one shared action expert is
in-distribution under both conditioning distributions:

```bash
python scripts/precompute_text_embeds.py task=libero_dual_regime_fused_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=libero_dual_regime_fused_2cam224_1e-4
```

For a storage-free diagnostic of the real training path, set both
`save_every=0` and `save_final_checkpoint=false` together with an explicit
small `max_steps`. `save_final_checkpoint` defaults to `true`, so normal runs
retain the existing final weights/state behavior. A smoke run is not a pilot
completion artifact and must not be passed to `validate_sdr_checkpoint.py` or
used to create `pilot_selection.json`.

The validated sequence and evidence are indexed under
[`docs/validation/e1/`](validation/e1/README.md): real-Wan P0 parity first,
standalone E-I to fused S0 forced-IDM parity second, and a one-step real LIBERO
forward/backward/optimizer smoke last. The two full learning-rate pilots remain
pending.

The action objective is

```text
L_action = (L_idm + w_uncond * L_uncond) / (1 + w_uncond)
```

so adding the second regime does not silently double the action/video gradient
ratio. Configure `train.action_regime_weight_uncond`; model factories require a
finite, strictly positive value so provenance cannot certify an untrained
UNCOND branch. The fused equivalence additionally requires temporal
`patch_size[0] == 1`, a clean timestep-zero first frame, token-wise timestep
modulation and an attention mode that isolates frame zero
(`first_frame_causal` or `per_frame_causal`). These are checked at runtime.

New weight files record a checkpoint id, model class, dimensions,
`adaptive_regimes=[uncond,idm]` and `adaptive_backbone_kind=idm`. Adaptive
inference also records the training `dataset_stats.json` SHA256 and positive
UNCOND training weight. The adapter rejects unloaded models and mismatched live
classes or metadata. `allow_unloaded_model=True` is development/test-only; old
weights require the explicit `allow_legacy_checkpoint=True` escape hatch after
manual verification.

## Cached gate feature

The VAE is run once per decision:

```python
state = adapter.encode_world_state(input_image)
decision_input = state.world_feat
result = adapter.act(
    input_image=input_image,
    encoded_state=state,
    mode=mode,
    context=context,
    context_mask=context_mask,
)
```

`EncodedWorldState.first_frame_latents` remains on the WAM device/dtype and is
passed directly into either inference path. `world_feat` is fixed-size float32:
a `1x2x2` adaptive spatial pool (`4*C`) plus per-channel spatial standard
deviation (`C`), or `5*C` total. It retains coarse camera layout rather than
collapsing the latent to one global mean. Text is represented separately by
`pool_text_context`: masked token mean followed by deterministic adaptive
average pooling to 64 dimensions. Oracle BC and online RL use the same helper.

`FastWAMIDM.infer_action` sets `decode_video=False`; adaptive action inference
never pays for pixel decoding. With a fixed seed, UNCOND and IDM share exactly
the same initial action-noise stream, while IDM video noise uses a deterministic
derived seed and is therefore independent.

## Compute cost

Reward cost is normalized to `cost(IDM)=1`. Profile the exact checkpoint and
deployment shape:

```bash
python scripts/profile_wam_modes.py \
  --task libero_dual_regime_fused_2cam224_1e-4 \
  --backbone-kind idm --ckpt /path/to/checkpoint.pt \
  --inference-steps 20 --height 224 --width 448 \
  --num-video-frames 9 --action-horizon 32 \
  --out configs/adaptive_gate/wam_cost_libero_idm.yaml
```

Both measurements include the one shared VAE encode. Loading an explicit cost
path is strict: missing files, invalid keys, non-finite/non-monotonic values,
unavailable raw sources, or mismatched task/checkpoint/backbone/steps/frames/
horizon/resolution fail instead of silently changing the RL objective.
Profiles additionally bind context length, model dtype and proprio dimension;
latency rewards bind the accelerator model name.
`source=latency` aliases the profiler's `latency_ms` field.

## Offline oracle analysis

The optional oracle runs both modes with paired action seeds and stores complete
per-step L1/L2 curves. Its corrected label is the cheapest measured-cost mode
within tolerance of the **best observed mode**, rather than treating IDM as an
infallible reference. Every record also stores `best_err`, absolute `idm_err`
and `idm_regret`, allowing BC to filter states where the expensive model itself
is poor. Samples with any non-finite per-step or chunk error are dropped and
counted as `num_nonfinite_dropped`; they are never silently labeled IDM.

Shards include the 64-D text feature and `group_id` (dataset id plus episode id)
for leakage-free train/validation splits without cross-dataset id collisions. Version-2 shards carry a strict
compatibility fingerprint over checkpoint, dataset stats, task, inference,
feature and relabeling settings; three-mode or incompatible shards must be
regenerated and cannot be concatenated. Loading also rejects duplicate or
incomplete shard-index sets unless `allow_partial=True` is explicitly requested.

```bash
python scripts/generate_gate_oracle_labels.py \
  --task libero_dual_regime_fused_2cam224_1e-4 \
  --backbone-kind idm --ckpt /path/to/checkpoint.pt \
  --dataset-stats /path/to/dataset_stats.json \
  --exec-horizon 10 --out data/gate_oracle/libero_idm
```

Padded tails are retained by default and removed by `valid_steps`; use
`--skip-padded` only for a deliberate complete-window subset. Oracle labels use
dataset ground-truth actions and are optional analysis/BC data, not part of the
default zero-supervision GRPO path.

## Routing ownership

Adaptive model inference requires explicit `force_branch` by default. The old
four-probe KDE entropy router is retained only behind
`allow_internal_routing=True` as a legacy ablation; because it performs repeated
action probes before its final branch, it is not a compute-saving mechanism.

Standalone LIBERO/RoboTwin evaluation reads `EVALUATION.force_branch` (default
`base`, i.e. UNCOND). Future-video visualization requires `idm`. Trainer video
validation always forces the model's complete future branch so periodic
`eval_every` remains valid while external routing is mandatory.
