# HiPerViT — Hybrid Multi-Scale Perception Vision Transformer

HiPerViT is a hybrid CNN–Transformer architecture for binary classification of dermoscopy images under severe class imbalance. It combines a pretrained **EfficientNet-B0** backbone with a shared **Vision Transformer (ViT)** encoder that operates simultaneously over three resolution scales. A novel **SkewMod** module injects a dataset-aware imbalance signal into the early Transformer layers, amplifying minority-class gradient flow during training without oversampling or loss re-weighting.

---

## Architecture

```
Input Image (B, C, H, W)
        │
        ▼
┌─────────────────────────┐
│  EfficientNet-B0        │  ← partial fine-tuning (last 3 blocks)
│  features_only=True     │
│  out_indices = [2,3,4]  │
└────────┬────────────────┘
         │  3 feature maps
    ┌────┴─────┬──────────┐
    ▼          ▼          ▼
 Scale-112  Scale-56   Scale-28
 Patchify   Patchify   Patchify
 + Project  + Project  + Project
    │          │          │
    └────┬─────┘          │
         │  concat        │
         ▼                │
  ┌──────────────────────────────────┐
  │   Shared Transformer Encoder     │
  │   + SkewMod (early 1/3 layers)   │
  └──────────────────────────────────┘
         │    │    │    │
       All  112   56   28   ← 4 independent CLS tokens
         └────┴────┴────┘
                │
         Concat (4 × dim)
                │
        ┌───────┴───────┐
        ▼               ▼
   MLP Head Con    MLP Head 28
   (primary)       (auxiliary)
        └───────┬───────┘
                ▼
          Final Logits
```

**Four resolution streams** are encoded independently through the same Transformer weights:

| Stream | Source feature map | Purpose |
|---|---|---|
| All-scales | 112 ‖ 56 ‖ 28 concatenated | Global multi-scale context |
| Scale-112 | EfficientNet stage 2 | Fine local detail |
| Scale-56 | EfficientNet stage 3 | Mid-level features |
| Scale-28 | EfficientNet stage 4 | High-level semantics |

The four CLS tokens are fused by the primary MLP head. The Scale-28 CLS token also feeds a dedicated auxiliary head (deep supervision). Final output is the **sum** of both heads.

---

## SkewMod — Imbalance-Aware Embedding Modulation

SkewMod addresses class imbalance directly inside the Transformer, rather than at the data or loss level.

At each of the first `floor(depth / 3)` Transformer layers, after the standard attention + FFN block:

1. A global average pool summarises the current token sequence → shape `(B, D)`
2. The pre-computed scalar `ρ = log(pos / neg)` is concatenated → shape `(B, D+1)`
3. A two-layer MLP maps this to a bias vector `(B, 1, D)`
4. The bias is broadcast and added back to all tokens

`ρ` is computed from the **actual training-split class distribution** before model construction (see `train.py: calculate_imbalance_ratio_rho`) and stored as a non-trainable buffer inside `SkewMod`, so it is saved with the model state-dict and restored on inference without any manual configuration.

---

## Repository Layout

```
HiPerViT/
│
├── models/
│   └── hipervit.py          # HiPerViT, Transformer, SkewMod, GradProbe
│
├── datasets/
│   ├── dataset_isic2017.py
│   ├── dataset_isic2018.py
│   ├── dataset_isic2024.py
│   ├── dataset_cbd4905.py
│   └── dataset_derm7pt.py
│
├── configs/
│   └── architecture.yaml    # Model hyperparameters
│
├── weights/                 # Saved checkpoints (created at runtime)
├── logs/                    # Training logs (created at runtime)
│
├── train.py                 # Training entry-point
├── predict.py               # Inference / evaluation entry-point
├── utils.py                 # Transforms, TTA helpers, loss utilities
├── early_stopping.py        # AUC-monitored early stopping
└── README.md
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA-capable GPU (recommended: ≥ 16 GB VRAM for batch size 32)

Install dependencies:

```bash
pip install torch torchvision timm einops \
            scikit-learn pandas numpy tqdm pyyaml \
            warmup-scheduler h5py
```

---

## Configuration

Model hyperparameters are defined in `configs/architecture.yaml`:

```yaml
model:
  image-size: 224
  patch-size: 4
  dim: 256
  depth: 6
  heads: 8
  dim-head: 32
  mlp-dim: 512
  emb-dim: 1024      # max positional embedding sequence length
  dropout: 0.1
  emb-dropout: 0.1
```

`emb-dim` must be ≥ the longest token sequence seen during training, which is `1 + N_112 + N_56 + N_28` (CLS token + all patches from the three scales concatenated).

---

## Datasets

### Supported datasets

| Key | Description | Image source |
|---|---|---|
| `ISIC2017` | ISIC 2017 Challenge — melanoma vs. non-melanoma | Image folder |
| `ISIC2018` | ISIC 2018 Challenge | Image folder |
| `ISIC2024` | ISIC 2024 Challenge | HDF5 file (`train-image.hdf5`) |
| `CBD4905` | Custom balanced skin lesion dataset | Image folder |
| `IMBD9810` | CBD4905 with 98 : 10 class imbalance split | Image folder |
| `Derm7pt` | Derm7pt dermoscopy dataset | Image folder |

---

### Common dataset module interface

Every dataset module follows the same contract so that `train.py` and
`predict.py` can work with any of them interchangeably.

**`get_df(data_dir)` / `get_test_df(data_dir)`**

Each module exposes one or both of these functions:

```python
df_train, df_valid, mel_idx = dataset_<name>.get_df(data_dir)
df_train, df_test,  mel_idx = dataset_<name>.get_test_df(data_dir)
```

- `df_train` / `df_valid` / `df_test` — pandas DataFrames with at minimum
  two columns: one for the image file path and one for the integer class label.
- `mel_idx` — integer index of the melanoma (positive) class used for
  AUC computation (typically `1`).

**Dataset class**

Each module also provides a `torch.utils.data.Dataset` subclass:

```python
dataset = <Name>_Dataset(df, split, transform=transforms)
# split: one of 'train', 'valid', 'test'
```

`__getitem__` returns `(image_id, image_tensor, label)` so that prediction
outputs can be traced back to individual samples.

---

### Expected directory structure

Place each dataset under its own root directory and pass that path as
`--data-dir`.  The layout expected by each module is shown below.

#### ISIC 2017

```
isic2017/
├── train_set/
│   └── *.jpg
├── valid_set/
│   └── *.jpg
├── test_set/
│   └── *.jpg
├── train_set_labels.csv
├── valid_set_labels.csv
└── test_set_labels.csv
```

CSV columns required:

| Column | Description |
|---|---|
| `image_id` | Filename without extension (e.g. `ISIC_0024306`) |
| `melanoma` | Binary label — `1` = melanoma, `0` = non-melanoma |

The dataset module constructs the full image path as:
`<data_dir>/<split>_set/<image_id>.jpg`

---

#### ISIC 2018

```
isic2018/
├── train_set/
│   └── *.jpg
├── valid_set/
│   └── *.jpg
├── test_set/
│   └── *.jpg
├── train_set_labels.csv
├── valid_set_labels.csv
└── test_set_labels.csv
```

CSV columns required:

| Column | Description |
|---|---|
| `image_id` | Filename without extension |
| `target` | Binary label — `1` = melanoma, `0` = non-melanoma |

---

#### ISIC 2024

```
isic2024/
├── train-image.hdf5    # all images stored inside this file keyed by isic_id
└── train-metadata.csv
```

CSV columns required:

| Column | Description |
|---|---|
| `isic_id` | Image key inside the HDF5 file |
| `target` | Binary label — `1` = melanoma, `0` = non-melanoma |

Images are read directly from the HDF5 file at runtime.  The path to the
HDF5 file is constructed as `<data_dir>/train-image.hdf5` and does not need
to be passed separately.

---

#### CBD4905 / IMBD9810

```
cbd4905/
├── images/
│   └── *.jpg
├── train_set.csv
├── valid_set.csv
└── test_set.csv
```

CSV columns required:

| Column | Description |
|---|---|
| `image_id` | Filename without extension |
| `target` | Binary label — `1` = melanoma, `0` = non-melanoma |

The dataset module constructs the full image path as:
`<data_dir>/images/<image_id>.jpg`

`IMBD9810` uses the same module and directory layout as `CBD4905`; the
imbalanced split is produced by `get_df_imbalanced_9810(data_dir)`.

---

#### Derm7pt

```
derm7pt/
├── images/
│   └── *.jpg
├── train_set.csv
├── valid_set.csv
└── test_set.csv
```

CSV columns required:

| Column | Description |
|---|---|
| `derm` | Dermoscopy image filename (with extension) |
| `diagnosis` | String diagnosis label (mapped to integer internally) |
| `target` | Binary label — `1` = melanoma, `0` = non-melanoma |

The dataset module constructs the full image path as:
`<data_dir>/images/<derm>`

---

### Adapting to a new dataset

To add a new dataset, create `datasets/dataset_<name>.py` following this
template:

```python
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def get_df(data_dir: str):
    """
    Load train/validation splits.

    Returns:
        df_train (DataFrame): Must contain 'filepath' and 'target' columns.
        df_valid (DataFrame): Same schema as df_train.
        mel_idx  (int):       Class index of the positive (melanoma) class.
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'))
    df_valid = pd.read_csv(os.path.join(data_dir, 'valid_set.csv'))

    # --- Adjust column names to match your CSV ----------------------------
    # 'image_id' → the image filename stem or full filename
    # 'target'   → integer binary label (1 = positive class)
    df_train['filepath'] = df_train['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', f'{x}.jpg')
    )
    df_valid['filepath'] = df_valid['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', f'{x}.jpg')
    )

    mel_idx = 1   # index of the melanoma / positive class
    return df_train, df_valid, mel_idx


def get_test_df(data_dir: str):
    """Load the held-out test split.  Same schema as get_df."""
    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'))
    df_test  = pd.read_csv(os.path.join(data_dir, 'test_set.csv'))

    df_test['filepath'] = df_test['image_id'].apply(
        lambda x: os.path.join(data_dir, 'images', f'{x}.jpg')
    )

    mel_idx = 1
    return df_train, df_test, mel_idx


class MyDataset(Dataset):
    """
    Args:
        df:        DataFrame with 'filepath' and 'target' columns.
        split:     One of 'train', 'valid', 'test'.
        transform: Albumentations or torchvision transform pipeline.
    """

    def __init__(self, df: pd.DataFrame, split: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.split     = split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row      = self.df.iloc[index]
        image_id = row['image_id']
        label    = int(row['target'])

        image = cv2.imread(row['filepath'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            result = self.transform(image=image)
            image  = result['image']

        return image_id, torch.tensor(image).float(), torch.tensor(label)
```

The only two things that must match your CSV are:

- **Image path column** — the column whose value, after any joining with
  `data_dir`, resolves to a readable image file.
- **Label column** — an integer column named `target` where `1` is the
  minority / positive class and `0` is the majority / negative class.

---

## Training

```bash
python train.py \
    --save-name      hipervit_isic2017 \
    --data-dir       /data/isic2017 \
    --dataset        ISIC2017 \
    --image-size     224 \
    --enet-type      efficientnet_b0 \
    --pretrained \
    --batch-size     32 \
    --n-epochs       30 \
    --init-lr        3e-5 \
    --config         ./configs/architecture.yaml \
    --model-dir      ./weights \
    --log-dir        ./logs
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--save-name` | required | Checkpoint / log base name |
| `--dataset` | required | One of `ISIC2017`, `ISIC2024`, `CBD4905`, `IMBD9810`, `Derm7pt` |
| `--image-size` | required | Square input resolution (e.g. `224`) |
| `--enet-type` | required | EfficientNet key: `efficientnet_b0` (1280ch) or `efficientnet_b7` (2560ch) |
| `--pretrained` | flag | Use ImageNet-pretrained backbone weights |
| `--batch-size` | `32` | Training batch size |
| `--n-epochs` | `20` | Maximum epochs (early stopping may end sooner) |
| `--init-lr` | `3e-5` | Adam initial learning rate |
| `--CUDA_VISIBLE_DEVICES` | `0` | GPU index to use |
| `--gpu-gc` | flag | Clear GPU cache before training |
| `--DEBUG` | flag | 2-epoch run on 150 train / 30 val samples |

The imbalance ratio `ρ` is computed automatically from the training split and requires no manual input.

**Scheduler:** 1-epoch linear warmup (×10 multiplier) followed by cosine-annealing warm restarts over the remaining epochs.

**Checkpoints saved:**

```
weights/<save-name>_epoch_N.pth   # every epoch
weights/<save-name>_best.pth      # best validation AUC
weights/<save-name>_final.pth     # last epoch
```

**Logs written:**

```
logs/log_<save-name>.txt    # plain-text per-epoch summary
logs/log_<save-name>.csv    # structured CSV for plotting
```

---

## Inference / Evaluation

```bash
python predict.py \
    --kernel-type    hipervit_isic2017 \
    --data-dir       /data/isic2017 \
    --dataset        ISIC2017 \
    --image-size     224 \
    --enet-type      efficientnet_b0 \
    --pretrained \
    --eval           best \
    --n-test         8 \
    --config         ./configs/architecture.yaml \
    --model-dir      ./weights \
    --sub-dir        ./subs
```

**Key inference arguments:**

| Argument | Default | Description |
|---|---|---|
| `--kernel-type` | required | Checkpoint base name (matches `--save-name` from training) |
| `--eval` | `best` | Load `_best.pth` or `_final.pth` |
| `--n-test` | `8` | Number of test-time augmentation (TTA) rounds |
| `--dataset` | required | One of `ISIC2017`, `ISIC2018`, `ISIC2024`, `Derm7pt` |

**Metrics reported:**

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix (TN, FP, FN, TP)
- Per-class accuracy and mean accuracy
- Full `sklearn` classification report

Results are printed to the console and appended to `subs/pred_<kernel-type>.txt`.

---

## GradProbe — Gradient Diagnostic Tool

`GradProbe` is a backward-hook utility included in `models/hipervit.py` for verifying that SkewMod successfully amplifies minority-class gradient norms during training.

```python
from models.hipervit import GradProbe

probe = GradProbe()
model.transformer.skew_mod.register_full_backward_hook(probe.hook)

for data, target in train_loader:
    probe.set_labels(target)       # must be called before each forward pass
    logits = model(data)
    loss = criterion(logits, target)
    loss.backward()

# Inspect per-batch gradient norms split by class
print(probe.logs[-1])              # {"g_min": float, "g_maj": float}
```

A healthy SkewMod should show `g_min > g_maj` (minority gradients amplified relative to majority).

---

## Multi-GPU Training

Multi-GPU training via `nn.DataParallel` is enabled automatically when more than one GPU is detected:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ...
```

When loading a DataParallel checkpoint with `predict.py` on a single GPU, the `module.` prefix is stripped automatically.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you use HiPerViT in your research, please cite:

```bibtex
@misc{hipervit2025,
  title  = {HiPerViT: Hybrid Multi-Scale Perception Vision Transformer
            with Skew Modulation for Imbalanced Skin Lesion Classification},
  author = {Ahammed, S. and others},
  year   = {2025},
  url    = {https://github.com/<your-username>/hipervit}
}
```
