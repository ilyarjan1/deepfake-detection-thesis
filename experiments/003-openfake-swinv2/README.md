# Experiment 003 — Released GenImage SwinV2 on OpenFake v1

Status: Completed on 2026-09-30.

## Objective

Evaluate the SwinV2 detector released with the OpenFake paper on exactly the
same frozen OpenFake v1 images used for Experiment 002, then measure whether
its errors complement AIDE's errors.

This is the released **GenImage-trained SwinV2** checkpoint, not the
OpenFake-trained checkpoint. That choice keeps the comparison meaningful:
both AIDE and this SwinV2 baseline were trained on GenImage and evaluated
out-of-distribution on OpenFake.

## Reproducibility record

- Official code: `vicliv/OpenFake`
- Code commit: `af44d6d37f5081b32610baeda9dbc49787edd594`
- Evaluator: `experiments/scripts/eval.py`
- Base architecture: `microsoft/swinv2-small-patch4-window16-256`
- Released checkpoint: `genimage.safetensors`
- Checkpoint SHA-256: `12ff8b29341dad47ddf1035047de7d07e5b7b41283ed57291d2a5ae61718a123`
- Checkpoint size: 195,905,448 bytes
- Environment: Python 3.10.21, PyTorch 2.14.0+cu130,
  Transformers 5.17.0
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Batch size: 32
- Data-loader workers: 4
- Runtime: 12 minutes 49 seconds
- Evaluated images: 59,655
- Dataset composition: 29,826 real and 29,829 fake
- Decision threshold: unchanged at 0.5

The original download contained 59,658 images. Three unreadable real images
were excluded in Experiment 002, so both detectors were compared on the same
59,655 readable files.

## SwinV2 results

| Metric | Result |
| --- | ---: |
| Accuracy | 78.9707% |
| Balanced accuracy | 78.9718% |
| Fake precision | 99.9134% |
| Fake recall | 57.9939% |
| Real specificity | 99.9497% |
| Fake F1 | 73.3895% |
| ROC-AUC | 0.925500 |
| Average precision | 0.948828 |
| False positives | 15 |
| False negatives | 12,530 |
| True negatives | 29,811 |
| True positives | 17,299 |

SwinV2 is extremely conservative at the released 0.5 threshold. It almost
never labels a real image as fake, but misses approximately 42% of synthetic
images. Its ROC-AUC and average precision remain strong, suggesting that
cross-dataset score calibration and threshold choice contribute substantially
to its poor thresholded recall.

## Comparison with AIDE

| Method | Accuracy | Fake recall | Specificity | Fake F1 | ROC-AUC | AP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AIDE | 85.9341% | 73.9381% | 97.9313% | 84.0174% | 0.925030 | 0.944040 | 617 | 7,774 |
| SwinV2 | 78.9707% | 57.9939% | 99.9497% | 73.3895% | 0.925500 | 0.948828 | 15 | 12,530 |
| Equal probability average | 80.2263% | 60.5015% | 99.9531% | 75.3686% | 0.942504 | 0.959183 | 14 | 11,782 |
| OR decision fusion | 89.4024% | 80.9078% | 97.8978% | 88.4191% | 0.940532 | 0.957820 | 627 | 5,695 |

The equal probability average performs poorly at the unchanged 0.5 threshold
because the two detectors have very different score calibration. A simple OR
decision rule improves over AIDE, but it is only an exploratory, unoptimized
result and has not yet been validated on a held-out dataset.

## Error overlap

### All images

- Both correct: 44,419
- Only AIDE correct: 6,845
- Only SwinV2 correct: 2,691
- Both wrong: 5,700

### Fake images

- Both detect: 15,220
- AIDE only: 6,835
- SwinV2 only: 2,079
- Both miss: 5,695

### Real-image false positives

- AIDE only: 612
- SwinV2 only: 10
- Both detectors: 5

SwinV2 correctly detects 2,079 fake images that AIDE misses. This establishes
measurable error complementarity and supports investigating calibrated fusion.
However, 5,695 fake images are missed by both detectors and remain the main
unresolved failure set.

## Largest potential per-generator gains from OR fusion

| Generator | AIDE recall | SwinV2 recall | Either detects | Gain over AIDE |
| --- | ---: | ---: | ---: | ---: |
| Imagen 4.0 | 17.3% | 49.3% | 58.6% | +41.3 points |
| Flux.1 Schnell | 75.9% | 93.0% | 96.3% | +20.4 points |
| GPT Image 1 | 50.6% | 57.5% | 65.4% | +14.8 points |
| Flux Amateur Snapshot Photos | 84.5% | 90.8% | 98.6% | +14.1 points |
| HiDream I1 Full | 79.5% | 69.8% | 93.1% | +13.6 points |
| Recraft v2 | 29.79% | 17.38% | 43.26% | +13.48 points |

Imagen 4.0 is the clearest complementarity case. Conversely, both detectors
perform very poorly on Midjourney 6, Flux 1.1 Pro, Grok 2, DALL-E 3, Recraft
v3, and Aurora, so fusion between only these two models cannot resolve those
generator families.

## Interpretation and limitations

This experiment demonstrates a promising research direction, not a final
thesis contribution. The OR rule was inspected and evaluated on the same
dataset, so its improvement must be confirmed on an untouched validation or
test set. Threshold calibration, score normalization, learned fusion, and
robustness experiments should be designed using a separate development split.

The released repository currently documents newer OpenFake v2 split files.
This experiment intentionally used the frozen v1 test data from Experiment 002
to ensure an exact per-image comparison with AIDE. Therefore, these numbers
should not be described as an exact reproduction of the paper's v2 tables.

## Next steps

1. Preserve the raw SwinV2 predictions and compact comparison tables.
2. Inspect the 2,079 SwinV2-only detections and 5,695 shared misses.
3. Create a development/test protocol before tuning any fusion parameters.
4. Test calibration and threshold selection on development data only.
5. Evaluate the frozen fusion rule on another public dataset such as DDL.
