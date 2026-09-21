# Research Log

## 2026-09-21 — AIDE environment established

- Cloned the official AIDE implementation.
- Created the `aide` Conda environment with Python 3.10.21.
- Installed PyTorch 2.0.1 with CUDA 11.8.
- Confirmed CUDA inference on an NVIDIA GeForce RTX 3070 Ti Laptop GPU.
- Downloaded the official `GenImage_train.pth` checkpoint.
- Patched evaluation gathering so non-distributed single-GPU inference works.
- Downloaded the GenImage Stable Diffusion v1.4 test subset from a Hugging Face
  mirror: 6,000 real and 6,000 fake images.

## 2026-09-21 — Experiment 001 completed

AIDE was evaluated on all 12,000 Stable Diffusion v1.4 test images.

- Accuracy: 99.825%
- Average precision: 0.999984342917076
- Incorrect classifications: 21
- Runtime: approximately 25 minutes
- Peak reported GPU memory: 3,743 MB

This verifies the implementation and evaluation pipeline. Because Stable
Diffusion v1.4 is represented in GenImage, this is primarily an in-distribution
baseline and does not by itself establish generalization.

## 2026-09-21 — OpenFake selected next

OpenFake was selected because it contains newer generator families, metadata
for per-generator analysis, released source code, and released SwinV2 weights.

OpenFake is continuously updated. To reproduce the paper-era experiment, we
pin:

- OpenFake source-code tag: `v1`
- Source commit: `2e5fe229bfc9424b165455ad366f09e04c5e5591`
- Hugging Face dataset revision: `v1.0`
- Frozen test size: 59,658 images

The next experiments will first evaluate unchanged AIDE on this frozen split,
then reproduce OpenFake's released SwinV2 baseline.
