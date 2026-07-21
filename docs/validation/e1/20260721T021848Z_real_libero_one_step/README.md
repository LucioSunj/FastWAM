# Real LIBERO One-Step S-DR Training Smoke

Status: **PASS diagnostic**, not a formal E1 pilot.

The run exercised the production `train_zero1.sh` path on one
NVIDIA A800-SXM4-80GB with the fused dual-regime LIBERO task, real Wan2.2 5B
and ActionDiT assets, strict standalone-IDM warm start, BF16, and DeepSpeed
ZeRO-1. The data contract remained 33 frames at two-camera 224x448 resolution.
Only batch size and workload duration were reduced to one.

Recorded results:

- one successful optimizer step and `dual_regime_optimizer_steps=1`;
- total loss `0.0099`;
- IDM action loss `0.0005`;
- UNCOND action loss `0.0075`;
- video loss `0.0019`;
- peak monitored GPU memory `40,127 MiB`;
- wall clock `880.395 seconds`;
- zero files under the checkpoint tree.

The run exposed an NCCL shutdown warning because the production runtime did not
call `Accelerator.end_training()`. The code in the commit containing this
archive adds guaranteed `finally` cleanup and an exception-path regression
test. The real run itself predates that cleanup-only change; its forward,
backward, and optimizer result is unaffected.

`save_final_checkpoint` now defaults to `true`, preserving normal training.
This smoke explicitly set it to `false` together with `save_every=0`, so the
storage-limited server did not write a model or optimizer checkpoint.

Evidence:

- [`smoke_result.json`](smoke_result.json) is the structured fail-closed result.
- [`launcher.log`](launcher.log) contains the complete training log and timing.
- [`config.yaml`](config.yaml) is the configuration consumed by training.
- [`resolved_config.yaml`](resolved_config.yaml) is the pre-run composition.
- [`gpu_monitor.csv`](gpu_monitor.csv) records one-second GPU samples.
- [`code_diff.patch`](code_diff.patch) captures the code under test plus the
  post-run cleanup fix.

The committed text copies of the launcher log, nvidia-smi snapshots, and patch
strip trailing whitespace only so repository diff checks remain clean. Artifact
hashes embedded in `smoke_result.json` describe the untouched server-local
sources; `SHA256SUMS.txt` is authoritative for the committed archive copies.

This smoke does not establish full-schedule stability, convergence, resume or
checkpoint correctness, pilot selection, or endpoint quality.
