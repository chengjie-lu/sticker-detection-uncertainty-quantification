# Sticker detector

Model can be used in two formats: Onnx and ckpt.
Download the models [here](https://github.com/chengjie-lu/sticker-detection-uq/tree/main/checkpoints/retinanet_resnet50_fpn_v2_aug/version_0/checkpoints) and put them in `checkpoints\retinanet_resnet50_fpn_v2_aug\version_0\checkpoints`

Onnx model name: `retinanet_resnet50_fpn_v2_aug.onnx`
ckpt model name: `epoch=31-step=7712.ckpt`

## Requirements Installation

```sh
pip install -r requirements.txt
```

Please follow [this instruction](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation) to build and install torchvision from source.

## Execution

```sh
python detection_offline.py
```
