# Per-generator cross-generator comparison

Matched-FPR adapted threshold: `0.748789`. Both models produce 596 false positives on real images.

| Generator | Images | Original recall | Adapted recall (0.5) | Change (0.5) | Adapted recall (matched FPR) | Matched change | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| aurora-20-1-25 | 282 | 1.4% | 86.9% | +85.5% | 56.0% | +54.6% | improved |
| recraft-v2 | 282 | 29.8% | 90.1% | +60.3% | 61.3% | +31.6% | improved |
| imagen-4.0 | 1000 | 17.3% | 57.5% | +40.2% | 30.7% | +13.4% | improved |
| midjourney-6 | 1000 | 27.1% | 59.7% | +32.6% | 33.8% | +6.7% | improved |
| dalle-3 | 1000 | 34.6% | 63.1% | +28.5% | 37.2% | +2.6% | improved |
| ideogram-2.0 | 282 | 90.8% | 94.0% | +3.2% | 90.8% | -0.0% | unchanged |
| midjourney-7 | 1000 | 98.2% | 98.8% | +0.6% | 97.7% | -0.5% | degraded |
| gpt-image-1 | 1000 | 50.6% | 64.4% | +13.8% | 49.7% | -0.9% | degraded |
| halfmoon-4-4-25 | 190 | 97.4% | 97.4% | +0.0% | 95.3% | -2.1% | degraded |
| imagen-3.0-002 | 1000 | 93.4% | 95.3% | +1.9% | 90.8% | -2.6% | degraded |
| frames-23-1-25 | 250 | 96.0% | 97.6% | +1.6% | 93.2% | -2.8% | degraded |
| sd-3.5 | 1000 | 94.9% | 96.6% | +1.7% | 92.1% | -2.8% | degraded |
| flux-realism | 1000 | 89.5% | 91.1% | +1.6% | 86.7% | -2.8% | degraded |
| sd-1.5-epicdream | 1000 | 96.4% | 95.8% | -0.6% | 92.9% | -3.5% | degraded |
| grok-2-image-1212 | 1000 | 19.3% | 31.1% | +11.8% | 15.8% | -3.5% | degraded |
| chroma | 1000 | 83.5% | 83.4% | -0.1% | 79.1% | -4.4% | degraded |
| sdxl-touchofrealism | 1000 | 97.0% | 96.8% | -0.2% | 92.6% | -4.4% | degraded |
| lumina-17-2-25 | 543 | 95.6% | 95.4% | -0.2% | 90.8% | -4.8% | degraded |
| flux.1-dev | 1000 | 91.8% | 92.2% | +0.4% | 86.8% | -5.0% | degraded |
| sdxl-realvis-v5 | 1000 | 91.9% | 91.2% | -0.7% | 85.8% | -6.1% | degraded |
| sd-1.5-dreamshaper | 1000 | 95.3% | 94.1% | -1.2% | 88.9% | -6.4% | degraded |
| sdxl-juggernaut | 1000 | 93.6% | 93.9% | +0.3% | 87.2% | -6.4% | degraded |
| sdxl | 1000 | 94.3% | 92.9% | -1.4% | 87.8% | -6.5% | degraded |
| mystic | 1000 | 95.1% | 93.5% | -1.6% | 88.2% | -6.9% | degraded |
| ideogram-3.0 | 1000 | 88.0% | 87.7% | -0.3% | 80.3% | -7.7% | degraded |
| sd-2.1 | 1000 | 97.2% | 96.0% | -1.2% | 89.5% | -7.7% | degraded |
| hidream-i1-full | 1000 | 79.5% | 79.9% | +0.4% | 71.8% | -7.7% | degraded |
| flux-amateursnapshotphotos | 1000 | 84.5% | 83.2% | -1.3% | 76.7% | -7.8% | degraded |
| flux-mvc5000 | 1000 | 84.2% | 83.0% | -1.2% | 75.1% | -9.1% | degraded |
| flux.1-schnell | 1000 | 75.9% | 75.0% | -0.9% | 66.4% | -9.5% | degraded |
| flux-1.1-pro | 1000 | 22.9% | 30.5% | +7.6% | 12.6% | -10.3% | degraded |
| sd-1.5 | 1000 | 97.0% | 93.2% | -3.8% | 85.7% | -11.3% | degraded |
| sdxl-epic-realism | 1000 | 79.2% | 78.9% | -0.3% | 66.7% | -12.5% | degraded |
