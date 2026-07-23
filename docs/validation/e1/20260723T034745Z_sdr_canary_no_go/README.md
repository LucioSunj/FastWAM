# S-DR 50-Step Canary No-Go

This archive preserves the accepted no-update P0.5 baseline and the first
completed 50-optimizer-step shared dual-regime Canary for the user-attested
original-Wan LIBERO E-I checkpoint.

## Bound inputs

- E-I checkpoint:
  `/autodl-fs/data/fastwam/checkpoints/fastwam_idm_libero_step_043400.pt`
- E-I SHA256:
  `f6e29dd6638d19a9e60c87ab387f0cc8d5c75f1ab9262cacd8d269ea1ee43c9c`
- Training code commit: `1434fc914a51d9078aae546b53cc9e7610cc250d`
- P0.5 run: `20260723T033214Z`
- Canary run: `20260723T034745Z`
- Plus-Full outcomes were not loaded or used.

## Result

P0.5 passed with zero optimizer updates, selected
`w0=0.009558040980013546`, and authorized `w_cap=0.5`. The Canary then
completed exactly 50 successful optimizer steps from an independent E-I/S0
warm start. It had no skipped steps, no non-finite metrics, zero gradient
clipping, and a peak GPU allocation of 33,920,923,136 bytes.

The preregistered Canary decision is `FAIL-DIAGNOSED`. The only formal
failure condition is `common-noise IDM margin is not positive`:

| Metric | Step 0 | Step 50 |
|---|---:|---:|
| Common action-all IDM margin at w=0.5 | 0.0874747491 | -0.0903124652 |
| Final-block negative IDM-margin fraction | 0.125 | 0.75 |
| UNCOND raw loss | 0.2864014894 | 0.1739948267 |
| Generated-IDM normalized action L2 | 0.1068509231 | 0.1087805834 |
| Future-sensitivity median | 0.4067696333 | 0.4042735845 |

The generated-IDM error increase was about 1.81%, no-read/forced-UNCOND
parity remained exact, and future sensitivity was retained. These observations
do not override the negative IDM descent margin. The 500-step probe and
10-epoch formal training were not authorized and were not started.

## Checkpoint retention

The 2.0 GiB ActionDiT-only step-50 delta is intentionally not committed:

- Local path:
  `/root/autodl-tmp/experiments/adaptive_wm_reasoning/E1_P1/20260723T034745Z/canary/train/checkpoints/weights/step_000050.action_dit_delta.pt`
- SHA256:
  `fd0baa7fb5d7989ac0f6f67bf73c38b5f9596bede2bc0d63b8c2188efe763ca2`

The committed `canary_decision.json` binds that delta by path, size, and
SHA256. Raw baseline and step-50 gradient/generated-future diagnostics are
included so the decision can be independently audited.

## Tooling note

After the scientific run stopped, commit `ff12c447a6de8997b5e1babe979ec1d7da63f8ed`
fixed two diagnostic-orchestration issues without changing the trained delta:
logged schedule weights are compared at their actual IEEE float32 precision,
and reused P0.5 caches resolve to the provenance-verified original cache
creation. The full first-party suite then reported 161 passed and 22 skipped.
Any new training run must bind and rerun P0.5 on its own final code commit.
