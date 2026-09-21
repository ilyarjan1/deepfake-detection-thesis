# Deepfake Detection Thesis Research

Exploratory research repository for evaluating the generalization of recent
deepfake and AI-generated-image detectors.

The final thesis problem and proposed method have not yet been fixed. The
current stage reproduces published results, evaluates models on newer public
datasets, identifies systematic weaknesses, and records evidence that may
eventually motivate a new method.

## Current research question

How well does AIDE generalize from GenImage to newer and more realistic public
deepfake datasets, and under which generators, manipulation types, image
categories, or perturbations does it struggle?

## Experiment roadmap

| ID | Experiment | Status |
|---|---|---|
| 001 | AIDE on GenImage Stable Diffusion v1.4 | Completed |
| 002 | AIDE on frozen OpenFake v1 test set | Preparing |
| 003 | Reproduce OpenFake SwinV2 baseline | Planned |
| 004 | AIDE on DDL-I | Future |

## Repository policy

Large datasets, model checkpoints, archives, and Hugging Face caches are not
stored in GitHub. This repository stores download scripts, dataset revisions,
checksums, environment details, commands, metrics, and analysis needed to
reproduce the work.

Third-party projects remain in their original repositories and are referenced
by URL and exact commit.
