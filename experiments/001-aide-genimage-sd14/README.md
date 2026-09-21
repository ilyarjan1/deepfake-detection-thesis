# Experiment 001 — AIDE on GenImage SD v1.4

## Objective

Verify that the official AIDE implementation, released checkpoint, dataset
layout, preprocessing, and GPU inference pipeline work correctly.

## Software and hardware

- AIDE commit: `6725b710d5c437ab2f59792908ce0377dfc907de`
- Python: 3.10.21
- PyTorch: 2.0.1+cu118
- CUDA runtime: 11.8
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Checkpoint: `GenImage_train.pth`

## Dataset

- GenImage Stable Diffusion v1.4 test subset
- 6,000 real images
- 6,000 fake images
- 12,000 total images

## Results

| Metric | Result |
|---|---:|
| Accuracy | 99.825% |
| Average precision | 0.999984342917076 |
| Errors | 21 / 12,000 |
| Loss | 0.059 |
| Runtime | 25 minutes |
| Peak GPU memory | 3,743 MB |

## Interpretation

The result successfully validates the reproduction pipeline. It should not be
treated as evidence of broad out-of-distribution generalization because Stable
Diffusion v1.4 is included in the GenImage domain used by the checkpoint.

## Implementation note

The released evaluation function attempted `torch.distributed.all_gather`
during a non-distributed single-GPU run. The stored patch guards distributed
gathering and uses local predictions when no process group is initialized.
