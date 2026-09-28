# OpenFake v1 Test Manifest

- Hugging Face repository: `ComplexDataLab/OpenFake`
- Frozen revision: `v1.0`
- Revision commit: `750279a9710d3d98bbc6d0d87250312cfdf922a0`
- Split: `test`
- Expected examples: 59,658
- Approximate decoded size: 33.4 GB
- License: CC BY-NC 4.0

Images will be stored locally under `dataset/openfake_v1/` and excluded from
GitHub. The exporter preserves original image bytes to avoid introducing new
compression artifacts.

## Local integrity validation

The frozen OpenFake v1 test split originally contained 59,658 files:
29,829 real and 29,829 fake.

PIL integrity validation identified three unreadable real-image files. They were
excluded from evaluation because PIL could not identify their image format:

- Dataset index 15147: `0_real/015147.jpg`
- Dataset index 56476: `0_real/056476.jpg`
- Dataset index 59224: `0_real/059224.jpg`

Final evaluation set:

- Real: 29,826
- Fake: 29,829
- Total: 59,655

The exclusions represent approximately 0.005% of the original test set.
