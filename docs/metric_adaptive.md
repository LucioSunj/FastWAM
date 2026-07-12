# Dual-regime FastWAM

`MetricAdaptiveFastWAM` inherits `FastWAMIDM` and exposes two inference paths on
one shared checkpoint:

- `force_branch="base"`: reactive/current-frame inference, called `UNCOND` by
  the external gate.
- `force_branch="idm"`: complete two-stage IDM future/action inference.

The recommended model is the fused implementation and task config:

```bash
python scripts/precompute_text_embeds.py task=libero_dual_regime_fused_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=libero_dual_regime_fused_2cam224_1e-4
```

Every step trains the shared action expert under both conditioning regimes and
computes video loss once. The normalized objective is

```text
L = lambda_video * L_video
  + lambda_action * (L_idm + w_uncond * L_uncond) / (1 + w_uncond)
```

Configure `train.action_regime_weight_uncond`. The previous
`action_regime_weight_base` key is accepted with a warning for old configs. A
dual-regime factory requires a strictly positive UNCOND weight.

The external RL gate owns routing. `infer_action` therefore requires an explicit
`force_branch` unless `adaptive.allow_internal_routing=true` is deliberately set.
That flag exists only for legacy metric/selector experiments. In particular,
the PolicyEntropy/KDE router performs repeated action probes before running the
selected path and should not be reported as a compute-saving method.

Action-only UNCOND inference supports both `first_frame_causal` and
`per_frame_causal`; either isolates the sole encoded frame. Bidirectional masks
remain invalid for this contract.

New checkpoints include provenance identifying the dual regimes and IDM
backbone. See `docs/adaptive_gate.md` for the cached-latent API, strict cost
profiling, oracle analysis and RL integration contract.
