# E-I Step 43,400 User-Provided Training Metadata

The user supplied `config.yaml` and `dataset_stats.json` on 2026-07-22 and
identified them as training metadata associated with the standalone LIBERO
FastWAM-IDM checkpoint at step 43,400. The two files are preserved here
byte-for-byte; `SHA256SUMS.txt` records their attachment hashes.

## Bound checkpoint candidate

- Path: `/autodl-fs/data/fastwam/checkpoints/fastwam_idm_libero_step_043400.pt`
- SHA256: `f6e29dd6638d19a9e60c87ab387f0cc8d5c75f1ab9262cacd8d269ea1ee43c9c`
- Size: 12,041,735,601 bytes
- Recorded step: 43,400

## Supplied files

- `config.yaml`: SHA256
  `d81e530feb48392badb97cb44ab757778b3e3acdc4b2542f247db146c4f6b9d6`;
  4,716 bytes.
- `dataset_stats.json`: SHA256
  `4018617a46d738499288dde749f126c645c7617e4da938bd308d3670b3c35658`;
  40,949 bytes.

The config records a per-process batch size of 16, gradient accumulation 1,
learning rate `1e-4`, 10 epochs, BF16, seed 42, and
`mot_checkpoint_mixed_attn=true`. The stats record 1,712 episodes and 277,713
transitions. Their hash differs from the previously considered
`libero_uncond_2cam224_dataset_stats.json`, so the UNCOND-named candidate must
not be silently substituted for this file.

This archive upgrades the earlier config reconstruction with direct
user-provided metadata. It is not, by itself, a cryptographic proof that the
checkpoint was produced from these files: the original run manifest and
world-size/launcher record were not supplied with these two attachments. The
user subsequently clarified that this E-I was initialized from the original
`Wan-AI/Wan2.2-TI2V-5B` video/VAE components plus the recorded ActionDiT
initializer, not from Wan-Robot. A continuation manifest may therefore bind
the exact local parent components and record that parent relation as
user-attested, but must not relabel it as training-emitted provenance.
