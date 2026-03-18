# Mamba2D — Downstream Integrations (Detection & Segmentation)

## Prerequisites
- MMDetection ≥ 3.0 **or** MMSegmentation ≥ 1.0 with standard mmcv ≥ 2.0.0
- Pretrained ImageNet backbone checkpoints (M2D-N/T/S.ckpt) from [releases]
- Extra pip deps: `timm einops ninja pybind11`
  (CUDA wavefront kernel is JIT-compiled on first run)

## Setup

1. Copy `Mamba2D/` to your mm* repo:
   - Detection:    `mmdetection/projects/M2D/Mamba2D/`
   - Segmentation: `mmsegmentation/projects/M2D/Mamba2D/`

2. Copy configs:
   - Detection:    `configs/detection/` → `mmdetection/projects/M2D/configs/`
   - Segmentation: `configs/segmentation/` → `mmsegmentation/projects/M2D/configs/`

3. In both `tools/train.py` and `tools/test.py`, add near the top:
   ```python
   import sys; sys.path.insert(0, 'projects')
   ```

4. Place pretrained backbone checkpoints in `projects/M2D/pretrained/` (download from releases).
   The configs already point to `projects/M2D/pretrained/<name>.ckpt`.

## Training

Detection (Mask R-CNN, COCO):
```bash
python tools/train.py projects/M2D/configs/mask-rcnn_M2D-T_fpn_1x_coco.py
```

Segmentation (UperNet, ADE20K 160k):
```bash
python tools/train.py projects/M2D/configs/upernet_M2D-T_ade20k_160k_512x512.py
```

## Evaluation

Detection:
```bash
python tools/test.py projects/M2D/configs/mask-rcnn_M2D-T_fpn_1x_coco.py \
    path/to/checkpoint.pth
```

Segmentation — single-scale (SS):
```bash
python tools/test.py projects/M2D/configs/upernet_M2D-T_ade20k_160k_512x512.py \
    path/to/checkpoint.pth
```

Segmentation — multi-scale (MS), matches paper MS mIoU numbers:
```bash
python tools/test.py projects/M2D/configs/upernet_M2D-T_ade20k_160k_512x512.py \
    path/to/checkpoint.pth --tta
```

