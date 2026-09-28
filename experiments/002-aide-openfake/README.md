# Experiment 002 — AIDE on OpenFake v1

Status: Completed baseline evaluation and generator-level analysis.

## Objective

Evaluate the unchanged AIDE detector outside its GenImage training distribution
and identify modern image-generator families on which it fails.

## Model

- Repository: [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE)
- Repository commit: `6725b710d5c437ab2f59792908ce0377dfc907de`
- Model: AIDE
- Checkpoint: `GenImage_train.pth`
- Checkpoint training distribution: GenImage
- Classification threshold: 0.5, unchanged
- Local execution: one NVIDIA RTX 3070 Ti Laptop GPU
- Batch size: 1

Two local evaluation-only changes were used:

1. Skip distributed `all_gather` when no process group is initialized.
2. Export each image path, true label, fake probability, predicted label, and
   correctness.

Both patches are stored in this experiment directory.

## Dataset

- Dataset: [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake)
- Frozen revision: `v1.0`
- Revision commit: `750279a9710d3d98bbc6d0d87250312cfdf922a0`
- Split: `test`
- Original size: 59,658 images
- Valid evaluation size: 59,655 images
- Real: 29,826
- Fake: 29,829

Three unreadable real files were excluded before evaluation. Their indices and
paths are recorded in `excluded_images.csv`.

## Evaluation command

```bash
CUDA_VISIBLE_DEVICES=0 \
systemd-inhibit --what=sleep \
  --why="Running AIDE evaluation on OpenFake v1" \
  python main_finetune.py \
    --model AIDE \
    --batch_size 1 \
    --data_path dataset/smoke \
    --eval_data_path dataset/openfake_eval \
    --resume pretrained_ckpts/GenImage_train.pth \
    --eval True \
    --dist_eval False \
    --num_workers 0 \
    --output_dir results/openfake_v1
```

The clean evaluation wrapper `dataset/openfake_eval/OpenFake` was a symbolic
link to the locally exported OpenFake image directory. It prevented AIDE from
interpreting metadata CSV files as dataset folders.

## Results

| Metric | Result |
| --- | ---: |
| Accuracy | 85.9341% |
| Balanced accuracy | 85.9347% |
| Fake precision | 97.2786% |
| Fake recall | 73.9381% |
| Real specificity | 97.9313% |
| Fake F1 | 84.0174% |
| ROC-AUC | 0.925030 |
| Average precision | 0.944040 |
| False positives | 617 |
| False negatives | 7,774 |

Runtime was 5:36:29 with peak reported GPU memory of 3,728 MB.

## Main finding

AIDE is conservative on OpenFake. It preserves high specificity on real images
but misses 7,774 fake images. The errors are generator-dependent rather than
uniform.

Lowest fake recall:

| Generator | Fake recall | Missed / total |
| --- | ---: | ---: |
| Aurora | 1.42% | 278 / 282 |
| Recraft v3 | 4.50% | 955 / 1,000 |
| Imagen 4 | 17.30% | 827 / 1,000 |
| Grok 2 | 19.30% | 807 / 1,000 |
| Flux 1.1 Pro | 22.90% | 771 / 1,000 |
| Midjourney 6 | 27.10% | 729 / 1,000 |
| Recraft v2 | 29.79% | 198 / 282 |
| DALL-E 3 | 34.60% | 654 / 1,000 |
| GPT Image 1 | 50.60% | 494 / 1,000 |

By OpenFake metadata type, fake recall was 65.07% for base models, 88.80% for
LoRA variants, 91.92% for fine-tuned variants, and 93.60% for base-refined
variants. The base-model category contains most of the severe failures.

## Comparison with the OpenFake baseline

The OpenFake paper reports that its GenImage-trained SwinV2 reaches F1 0.881,
ROC-AUC 0.926, PR-AUC 0.949, real-image TNR 0.955, and average per-generator
TPR 0.823 on OpenFake. AIDE has similar ranking metrics but lower fake recall
and F1 while preserving higher real-image specificity.

The detectors also have different generator-specific failure patterns. This
makes complementary fusion worth testing, but the current evidence does not
yet establish that a particular fusion method will improve performance.

Paper: [OpenFake, arXiv:2509.09495](https://arxiv.org/abs/2509.09495)

## Interpretation and limitations

This is an out-of-distribution evaluation of one published checkpoint. It does
not show that AIDE is generally weak, and it does not by itself establish a
novel contribution. It shows a reproducible cross-generator generalization gap.

Release-date aggregates should be interpreted cautiously because several
OpenFake records use shared or approximate metadata dates. Generator-level
results are the stronger evidence.

## Next steps

1. Inspect confidently missed fake images and false-positive real images.
2. Test robustness to JPEG compression, resizing, blur, and screenshots.
3. Reproduce the OpenFake SwinV2 baseline on the same frozen split.
4. Compare per-image error overlap between AIDE and SwinV2.
5. Test score-level fusion only if the detectors make complementary errors.
6. Repeat the external evaluation on DDL-I.
