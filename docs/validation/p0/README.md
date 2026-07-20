# P0 Real-Wan Validation History

This directory preserves P0 evidence in chronological order. Historical
failures remain part of the audit trail and must not be overwritten by a later
passing run.

| UTC evidence/run | Implementation | Result | Evidence |
|---|---|---|---|
| 20260720T023208Z | Original dense single-MoT fused path | FAIL | [20260720T023208Z/](20260720T023208Z/) |
| 20260720T094723Z | Grouped-attention single-MoT path | FAIL due to /tmp storage exhaustion after 27 tests passed | [first grouped disk failure](20260720T100904Z_p0_grouped_repair_final/first_grouped_disk_failure/) |
| 20260720T100904Z | Grouped-attention single-MoT path | PASS, 28 passed and no skip | [final grouped PASS](20260720T100904Z_p0_grouped_repair_final/) |

The original dense fused implementation failed real-Wan UNCOND numerical
parity at the unchanged atol=5e-2, rtol=5e-2 acceptance threshold. The
grouped-attention repair retains one MoT.forward call while executing
reference-shaped attention groups within that call. Its first complete attempt
reached the final checkpoint-provenance test after all numerical parity cases
passed, then failed because PyTorch could not write the temporary checkpoint
under /tmp.

The final accepted retry redirected only pytest temporary storage to the
experiment filesystem. The task, test source, model, BF16 dtype, synthetic
shape, attention masks, solver, and tolerances remained unchanged. The
auto-generated Hydra output_dir timestamp necessarily changed between runs.

No E0, E1-S0, rollout, or training run is represented by these P0 artifacts.
