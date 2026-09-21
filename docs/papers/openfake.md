# OpenFake

**Paper:** OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection  
**Paper:** https://arxiv.org/abs/2509.09495  
**Code:** https://github.com/vicliv/OpenFake  
**Dataset:** https://huggingface.co/datasets/ComplexDataLab/OpenFake

## Relevance

OpenFake covers recent proprietary and open-source image generators and
provides generator-level metadata. This makes it suitable for identifying
specific families on which a GenImage-trained detector may fail.

## Reproducibility decision

OpenFake changes over time. Experiments corresponding to the original release
must use dataset revision `v1.0` and code tag `v1`. Results from the current
continually updated dataset must be reported separately.

## Planned work

1. Evaluate unchanged AIDE on the frozen test split.
2. Report overall and per-generator metrics.
3. Reproduce the official SwinV2 evaluation.
4. Compare AIDE and SwinV2 using the same images and labels.
