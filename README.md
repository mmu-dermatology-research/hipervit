# HiPerViT: A Hybrid Multi-Scale Encoder for Hierarchical Patch Representation on Imbalanced Low-Resolution Data

HiPerViT is a hybrid CNN–Transformer architecture for binary classification of dermoscopy images under severe class imbalance. It combines a pretrained **CNN** backbone with a shared **Vision Transformer (ViT)** encoder that operates simultaneously over three resolution scales. A novel **SkewMod** module injects a dataset-aware imbalance signal into the middle Transformer layers, amplifying minority-class gradient flow during training without oversampling or loss re-weighting.

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
├── datasets/                # dataset loader modules
│   ├── dataset_isic2017.py
│   ├── dataset_isic2018.py
│   ├── dataset_isic2024.py
│   ├── dataset_cbd4905.py
│   └── dataset_derm7pt.py
│
├── config.yaml              # Model hyperparameters
│
├── train.py                 # Training entry-point
├── predict.py               # Inference / evaluation entry-point
├── utils.py                 # Transforms, TTA helpers, loss utilities
├── early_stopping.py        # AUC-monitored early stopping
└── README.md
```

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
    --config         ./config.yaml \
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


## Citation

```

@inproceedings{ahammed2026hipervit,
  title={HiPerViT: A Hybrid Multi-Scale Encoder for Hierarchical Patch Representation on Imbalanced Low-Resolution Data},
  author={Ahammed, Sakib and Cui, Xia and Fan, Xinqi and Lu, Wenqi and Yap, Moi Hoon},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)},
  year={2026},
  organization={IEEE},
  note={In press}
}
```
