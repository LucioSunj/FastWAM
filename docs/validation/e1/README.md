# E1 Validation History

This directory preserves pre-pilot E1 evidence. It follows the accepted
[P0 real-Wan validation history](../p0/README.md): P0 first established that
the grouped fused implementation matches the reference two-forward model, then
E1-S0 checked that a standalone LIBERO IDM checkpoint imports into the shared
fused model without changing its forced-IDM action output.

| UTC run | Check | Result | Evidence |
|---|---|---|---|
| 20260720T162134Z | Standalone E-I to fused S0 forced-IDM parity | PASS under user-confirmed legacy config reconstruction; max_abs=0 | [S0 archive](20260720T162134Z_s0_reconstructed_config_clean/) |
| 20260721T021848Z | Real LIBERO fused S-DR training smoke | PASS diagnostic; one optimizer step, no checkpoint | [smoke archive](20260721T021848Z_real_libero_one_step/) |

The S0 run used sample 0, seed 0, BF16, 20 video/action solver steps, the
unchanged `atol=5e-4` and `rtol=5e-3`, and produced identical `[32, 7]` source
and target action tensors. Its contract decision binds `parity_result.json`.
All five repositories were clean in the accepted run manifest.

The historical E-I resolved config was not recovered. The user confirmed that
the checkpoint used the default `libero_idm_2cam224_1e-4` recipe with global
batch halved from 128 to 64. With 277,713 transitions and 10 epochs,
`ceil(277713 / 64) * 10 = 43,400`, matching the checkpoint step. This is a
documented reconstruction, not independent recovery of historical provenance.

The subsequent smoke run used the real fused task, real LIBERO 33-frame
two-camera 224x448 input, BF16, DeepSpeed ZeRO-1, and the accepted E-I warm
start. Batch size and maximum steps were deliberately limited to one. It
completed forward, backward, and one optimizer step with finite IDM, UNCOND,
and video losses, while writing zero checkpoint files.

Neither archive is an E1 pilot completion claim. The two preregistered S-DR
pilots, post-pilot selection, forced shared endpoints, controls, and downstream
E2-E6 experiments remain pending. E0 was skipped by explicit user direction;
no E0 PASS decision was created.
