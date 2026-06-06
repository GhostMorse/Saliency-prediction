# Video Saliency Prediction

Predicting visual **saliency / attention maps** for video frames in PyTorch:
given a frame, the model outputs a one-channel map of where a human viewer is
likely to look. Two models are provided — a **DeepLabV3-ResNet50** baseline
(transfer learning from semantic segmentation) and an **improved EfficientNet-B4
encoder-decoder** with ASPP, channel attention and skip connections. Models are
scored with the standard saliency metrics **SIM**, **CC** and **NSS**.

## Overview

Saliency prediction estimates the spatial distribution of human visual attention
over an image. Because annotated data is scarce, both models lean on
encoders pre-trained on ImageNet / segmentation and are fine-tuned to regress
the ground-truth saliency map. The dataset is small, so the focus is on transfer
learning, regularisation, augmentation and a multi-term loss rather than raw
model size.

## Models

### Baseline — DeepLabV3-ResNet50

A DeepLabV3 ResNet-50 backbone with its ASPP head produces stride-8 features; a
small decoder upsamples them back to full resolution and reduces them to a single
saliency channel (sigmoid output). Trained with the **KL-divergence** loss
between the predicted and ground-truth attention distributions.

### Improved — EfficientNet-B4 encoder-decoder

The improved model targets the baseline's weaknesses — limited localisation
detail and a single-scale decoder:

- **EfficientNet-B4** encoder (stronger, multi-scale features).
- **ASPP** bottleneck for a large multi-rate receptive field.
- **Channel attention** (CBAM-style) after each decoder stage.
- **Skip connections** from the encoder for sharper spatial detail.
- A **combined loss** (MSE + (1 − SSIM) + KL + (1 − Pearson)).
- **Augmentation** (flips + photometric), a **warmup + cosine** LR schedule,
  gradient accumulation, and early stopping.

## Project structure

```
video-saliency-prediction/
|- saliency/
|  |- data/
|  |  |- dataset.py        # SaliencyFrameDataset (OpenCV) + InfiniteSampler
|  |  |- augment.py        # AdvancedAugmentation (joint frame/map) + mixup
|  |  \- transforms.py     # ImageNet normalise / denormalise
|  |- models/
|  |  |- baseline.py       # SaliencyModel (DeepLabV3-ResNet50)
|  |  |- improved.py       # ImprovedSaliencyModel (EfficientNet-B4)
|  |  \- blocks.py         # ChannelAttention, SpatialAttention, ASPPModule
|  |- losses.py            # kld_loss, CombinedSaliencyLoss
|  |- metrics.py           # similarity (SIM), cc (CC), nss (NSS), calculate_metrics
|  |- engine.py            # train loop, EarlyStopping, WarmupCosineLR
|  |- evaluate.py          # SaliencyEvaluator (per-frame inference)
|  \- utils.py             # padding, normalize_map, seeding
|- train.py                # training entry point
|- run.py                  # predict on the test split + report SIM/CC/NSS
|- make_gif.py             # overlay predictions on a video as a GIF
|- notebooks/
|  \- saliency_pipeline.ipynb   # end-to-end: train -> predict -> score -> GIF
\- tests/                  # pytest suite (runs on CPU)
```

## Installation

```bash
git clone https://github.com/GhostMorse/video-saliency-prediction.git
cd video-saliency-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU (~12 GB) is recommended for training; evaluation and the tests run on CPU.

## Data

The dataset is a set of short video clips with per-frame ground-truth saliency
maps and fixation points. Unzip your local copy so the repository contains:

```
data/public_tests/
|- 01_test_file_input/
|  |- train/<video>/source.mp4
|  \- test/<video>/source.mp4
\- 01_test_file_gt/
   |- train/<video>/{NNNN.png, fixations/NNNN.png}
   \- test/<video>/{NNNN.png, fixations/NNNN.png}
```

where `source.mp4` is the input clip, `NNNN.png` is the ground-truth saliency for
frame *N* (1-indexed), and `fixations/NNNN.png` is the binary fixation map used
for NSS. Point `--data-root` (CLI) or `DATA_ROOT` (notebook) at the
`data/public_tests` folder. Frames are zero-padded to 216×384 and ImageNet
normalised internally.

## Usage

### Training

```bash
# baseline (KL-divergence loss)
python train.py --model baseline --data-root data/public_tests --epochs 5

# improved EfficientNet-B4 model (combined loss)
python train.py --model improved --data-root data/public_tests --epochs 30
```

The best checkpoint (lowest validation loss) is written to `saliency.pth`.

### Evaluation

```bash
# predict for every test video and print SIM / CC / NSS
python run.py data/public_tests --model improved --checkpoint saliency.pth
```

### Notebook

`notebooks/saliency_pipeline.ipynb` runs the whole pipeline end to end — build the
datasets, train, save `saliency.pth`, predict saliency for the test videos into
`outputs/`, compute the metrics, and render an overlay GIF.

### Qualitative GIF

```bash
python make_gif.py --video data/public_tests/01_test_file_input/test/0001/source.mp4 \
    --pred-dir outputs/0001 --output out.gif
```

## Metrics

Models are ranked by the average of the per-metric ranks of **SIM**, **CC** and
**NSS** (ties broken in that priority order):

```
Place = (Place_SIM + Place_CC + Place_NSS) / 3
```

SIM is the histogram intersection of the predicted and ground-truth maps, CC is
their Pearson correlation, and NSS is the mean normalised saliency at fixated
locations.

## Implementation notes

- **Residual to a distribution:** the KL loss normalises both maps to sum to one,
  so the model is trained to match an attention *distribution*, not raw pixels.
- **Augmentation safety:** only the horizontal flip is applied spatially (jointly
  to frame and map); photometric augmentations touch the frame only, so they
  never move the attention regions.
- **Padding never downscales:** frames smaller than 216×384 are centred and
  zero-padded; larger frames are left untouched.
- **Checkpoint selection:** the best model is chosen by validation loss.

## Findings

- **Transfer learning is essential here.** With few training videos, a
  segmentation-pretrained backbone gives the baseline a strong starting point;
  training such an encoder from scratch on this data would not be competitive.
- **Localisation detail is the baseline's main limit.** A single-scale decoder
  produces blurry maps, which the improved model addresses with skip connections
  (high-resolution encoder features), ASPP (multi-rate context) and channel
  attention (re-weighting informative channels).
- **A multi-term loss helps.** Optimising SSIM and correlation alongside MSE/KL
  aligns training more directly with the SIM/CC evaluation metrics than a single
  pixel-wise term, and tends to yield sharper, better-correlated maps.
- **Regularisation matters on a small set.** Dropout, augmentation and early
  stopping reduce over-fitting given the limited data.

After training on the full dataset, record the achieved SIM / CC / NSS here.

## Tests

```bash
pytest
```

Covers padding / normalisation, the SIM / CC / NSS metrics, the losses, and both
models' output shapes and value ranges (built without pretrained weights). All
tests run on CPU.

## License

Released under the [MIT License](LICENSE).
