# Sticker detector

Model can be used in two formats: Onnx and ckpt. 
Both models can be found in `checkpoints\retinanet_resnet50_fpn_v2_aug\version_0\checkpoints`

Onnx model name: `retinanet_resnet50_fpn_v2_aug.onnx`
ckpt model name: `epoch=31-step=7712.ckpt`

## Requirements Installation

```sh
pip install requirements.txt
```

## Execution

```sh
python detection_offline.py
```
