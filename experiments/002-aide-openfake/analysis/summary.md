# AIDE on OpenFake v1

## Overall results

- Images: 59,655
- Accuracy: 85.9341%
- Balanced accuracy: 85.9347%
- Fake precision: 97.2786%
- Fake recall: 73.9381%
- Real specificity: 97.9313%
- Fake F1: 84.0174%
- ROC-AUC: 0.925030
- Average precision: 0.944040
- False positives: 617
- False negatives: 7,774

All threshold-based metrics use the pretrained model's unchanged 0.5 decision threshold.

## Lowest fake recall by generator

| model | images | detected_fake | missed_fake | fake_recall | mean_fake_probability | median_fake_probability |
| --- | --- | --- | --- | --- | --- | --- |
| aurora-20-1-25 | 282 | 4 | 278 | 0.0142 | 0.0698 | 0.0526 |
| recraft-v3 | 1000 | 45 | 955 | 0.0450 | 0.1006 | 0.0518 |
| imagen-4.0 | 1000 | 173 | 827 | 0.1730 | 0.2212 | 0.0659 |
| grok-2-image-1212 | 1000 | 193 | 807 | 0.1930 | 0.2723 | 0.1560 |
| flux-1.1-pro | 1000 | 229 | 771 | 0.2290 | 0.2923 | 0.1800 |
| midjourney-6 | 1000 | 271 | 729 | 0.2710 | 0.3239 | 0.1929 |
| recraft-v2 | 282 | 84 | 198 | 0.2979 | 0.3341 | 0.1974 |
| dalle-3 | 1000 | 346 | 654 | 0.3460 | 0.3710 | 0.2170 |
| gpt-image-1 | 1000 | 506 | 494 | 0.5060 | 0.5143 | 0.5206 |
| flux.1-schnell | 1000 | 759 | 241 | 0.7590 | 0.7322 | 0.9310 |

## Highest fake recall by generator

| model | images | detected_fake | missed_fake | fake_recall | mean_fake_probability | median_fake_probability |
| --- | --- | --- | --- | --- | --- | --- |
| midjourney-7 | 1000 | 982 | 18 | 0.9820 | 0.9299 | 0.9466 |
| halfmoon-4-4-25 | 190 | 185 | 5 | 0.9737 | 0.9143 | 0.9440 |
| sd-2.1 | 1000 | 972 | 28 | 0.9720 | 0.9104 | 0.9459 |
| sd-1.5 | 1000 | 970 | 30 | 0.9700 | 0.9119 | 0.9448 |
| sdxl-touchofrealism | 1000 | 970 | 30 | 0.9700 | 0.9019 | 0.9424 |
| sd-1.5-epicdream | 1000 | 964 | 36 | 0.9640 | 0.8985 | 0.9403 |
| frames-23-1-25 | 250 | 240 | 10 | 0.9600 | 0.8979 | 0.9394 |
| lumina-17-2-25 | 543 | 519 | 24 | 0.9558 | 0.8982 | 0.9468 |
| sd-1.5-dreamshaper | 1000 | 953 | 47 | 0.9530 | 0.8986 | 0.9424 |
| mystic | 1000 | 951 | 49 | 0.9510 | 0.8938 | 0.9434 |
