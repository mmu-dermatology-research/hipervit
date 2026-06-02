"""
train.py
--------
Training entry-point for HiPerViT on dermatology image classification datasets.

Supported datasets
------------------
- ISIC2017
- ISIC2024
- CBD4905
- IMBD9810
- Derm7pt  (dermoscopy split)

Training strategy
-----------------
- Imbalance ratio rho = log(pos / neg) is computed from the training-split
  DataFrame before model construction and forwarded to HiPerViT, which
  passes it into SkewMod.  This makes the imbalance conditioning
  dataset-aware without any hard-coded constants.
- Adam optimiser with a 1-epoch gradual warmup followed by cosine-annealing
  restarts (GradualWarmupSchedulerV2).
- AUC-monitored early stopping (patience = 10 epochs).
- Best checkpoint saved on highest validation AUC; final checkpoint saved at
  the end of training.
- Per-epoch CSV and plain-text logs written to ``--log-dir``.

Usage example
-------------
    python train.py \\
        --save-name hipervit_isic2017 \\
        --data-dir /data/isic2017 \\
        --dataset ISIC2017 \\
        --image-size 224 \\
        --enet-type efficientnet_b0 \\
        --pretrained \\
        --batch-size 32 \\
        --n-epochs 30 \\
        --config ./configs/architecture.yaml
"""

import gc
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from warmup_scheduler import GradualWarmupScheduler

from utils import get_trans, get_transforms

import datasets.dataset_isic2017  as DatasetISIC2017
import datasets.dataset_isic2024  as DatasetISIC2024
import datasets.dataset_cbd4905   as DatasetCBD4905
import datasets.dataset_derm7pt   as DatasetDerm7pt

from datasets.dataset_isic2024 import ISIC2024_Dataset
from datasets.dataset_isic2017 import ISIC2017_Dataset
from datasets.dataset_cbd4905  import ISICCBD4905_Dataset
from datasets.dataset_derm7pt  import Derm7pt_Dataset

from models.hipervit import HiPerViT
from early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train HiPerViT on a dermatology image dataset."
    )
    parser.add_argument("--save-name",   type=str, required=True,
                        help="Base name used for saved model checkpoints and logs.")
    parser.add_argument("--data-dir",    type=str, default=".",
                        help="Root directory containing the dataset.")
    parser.add_argument("--image-size",  type=int, required=True,
                        help="Input image resolution (square).")
    parser.add_argument("--enet-type",   type=str, required=True,
                        help="EfficientNet variant key (e.g. 'efficientnet_b0').")
    parser.add_argument("--pretrained",  action="store_true",
                        help="Initialise EfficientNet backbone with ImageNet weights.")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--init-lr",     type=float, default=3e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--out-dim",     type=int, default=2,
                        help="Number of output classes.")
    parser.add_argument("--n-epochs",    type=int, default=20,
                        help="Maximum number of training epochs.")
    parser.add_argument("--use-extra",   action="store_true",
                        help="Include extra/external training data if available.")
    parser.add_argument("--gpu-gc",      action="store_true",
                        help="Run GPU garbage collection before training starts.")
    parser.add_argument("--DEBUG",       action="store_true",
                        help="Debug mode: 2 epochs, small data subset.")
    parser.add_argument("--model-dir",   type=str, default="./weights",
                        help="Directory for saving model checkpoints.")
    parser.add_argument("--log-dir",     type=str, default="./logs",
                        help="Directory for saving training logs.")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0",
                        help="GPU index to use when multiple GPUs are present.")
    parser.add_argument("--dataset",     type=str, required=True,
                        choices=["ISIC2017", "ISIC2024", "CBD4905", "IMBD9810", "Derm7pt"],
                        help="Dataset to train on.")
    parser.add_argument("--config",      type=str, default="./config.yaml",
                        help="Path to the YAML architecture configuration file.")
    parser.add_argument("--efficient_net", type=int, default=0,
                        help="EfficientNet version index (0 or 7, default: 0).")

    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: Integer seed value (default: 0).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Imbalance ratio
# ---------------------------------------------------------------------------

def calculate_imbalance_ratio_rho(df: pd.DataFrame) -> float:
    """
    Compute the binary imbalance ratio ``rho = log(pos / neg)`` from the
    training-split DataFrame.

    This scalar is passed to :class:`models.hipervit.SkewMod` via
    :class:`models.hipervit.HiPerViT` so that the imbalance conditioning
    signal is derived from the *actual* class distribution rather than a
    hard-coded constant.

    A negative value of ``rho`` (pos < neg) indicates minority-class
    scarcity; ``rho = 0`` corresponds to a balanced dataset.

    Args:
        df: Training-split DataFrame.  Must contain a ``'target'`` column
            with integer class labels (0 = majority, 1 = minority).

    Returns:
        Scalar float ``rho = log(pos / neg)``.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'target': [0]*974 + [1]*100})
        >>> rho = calculate_imbalance_ratio_rho(df)
        >>> print(f"rho = {rho:.4f}")   # rho = -2.2762
    """
    class_counts = df["target"].value_counts().sort_index()
    pos = int(class_counts.get(1, 1))
    neg = int(class_counts.get(0, 1))
    rho = float(np.log(pos / max(neg, 1)))
    print(f"[rho | binary] pos={pos}, neg={neg}, rho={rho:.4f}")
    return rho


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    Extended version of :class:`GradualWarmupScheduler` that correctly
    transitions into the after-scheduler once the warmup period is complete.

    During warmup the learning rate is scaled linearly from 0 (or
    ``base_lr / multiplier``) up to ``base_lr * multiplier``.  After warmup
    it delegates entirely to *after_scheduler*.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super().__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        return [
            base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader, optimizer) -> float:
    """
    Run one full training epoch.

    Iterates over *loader*, computes the loss with the global ``criterion``,
    performs backpropagation and an optimiser step, and returns the mean
    training loss for the epoch.

    Args:
        model:     The model being trained (set to ``train()`` mode internally).
        loader:    DataLoader yielding ``(data, target)`` batches.
        optimizer: PyTorch optimiser.

    Returns:
        Mean training loss over all batches in the epoch.
    """
    model.train()
    train_loss = []
    bar = tqdm(loader)

    for data, target in bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description("loss: %.5f, smth: %.5f" % (loss_np, smooth_loss))

    return float(np.mean(train_loss))


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def val_epoch(
    model: nn.Module,
    loader,
    mel_idx: int,
    n_test: int = 1,
    get_output: bool = False,
):
    """
    Run one full validation epoch with optional test-time augmentation (TTA).

    For each batch the model is called ``n_test`` times, each time with a
    different augmentation produced by :func:`utils.get_trans`.  Logits and
    probabilities are averaged across TTA rounds before computing metrics.

    Args:
        model:      The model to evaluate (set to ``eval()`` mode internally).
        loader:     DataLoader yielding ``(data, target)`` batches.
        mel_idx:    Class index treated as the positive (melanoma) class for
                    AUC computation.
        n_test:     Number of TTA rounds (1 = no TTA).
        get_output: If ``True``, return raw ``(LOGITS, PROBS)`` arrays instead
                    of scalar metrics.

    Returns:
        If *get_output* is ``False``: ``(val_loss, accuracy, auc)`` tuple.
        If *get_output* is ``True``:  ``(LOGITS, PROBS)`` numpy arrays.
    """
    model.eval()
    val_loss = []
    LOGITS, PROBS, TARGETS = [], [], []

    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)

            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            probs  = torch.zeros((data.shape[0], args.out_dim)).to(device)

            for I in range(n_test):
                l = model(get_trans(data, I))
                logits += l
                probs  += l.softmax(1)

            logits /= n_test
            probs  /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = float(np.mean(val_loss))
    LOGITS  = torch.cat(LOGITS).numpy()
    PROBS   = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS

    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.0
    auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    return val_loss, acc, auc


# ---------------------------------------------------------------------------
# Dataset construction and training orchestration
# ---------------------------------------------------------------------------

def run(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    transforms_train,
    transforms_val,
    mel_idx: int,
    rho: float,
) -> None:
    """
    Build datasets, instantiate HiPerViT with the pre-computed imbalance
    ratio, and run the full training loop.

    The *rho* value is computed from *df_train* by
    :func:`calculate_imbalance_ratio_rho` in :func:`main` and forwarded here
    so that :class:`models.hipervit.SkewMod` is conditioned on the real
    class distribution of the training split.

    Saves a per-epoch checkpoint, the best-AUC checkpoint, and the final
    checkpoint.  Logs are written to ``<log-dir>/log_<save-name>.txt`` and
    ``<log-dir>/log_<save-name>.csv``.

    Args:
        df_train:         Training split DataFrame.
        df_valid:         Validation split DataFrame.
        transforms_train: Albumentations / torchvision transform for training.
        transforms_val:   Albumentations / torchvision transform for validation.
        mel_idx:          Class index of the melanoma (positive) class.
        rho:              Imbalance ratio ``log(pos / neg)`` computed from
                          *df_train*.
    """
    # --- Debug mode: shrink data for fast iteration -----------------------
    if args.DEBUG:
        args.n_epochs = 2
        df_train = df_train.sample(150)
        df_valid = pd.concat([
            df_valid[df_valid["target"] == 0].sample(25),
            df_valid[df_valid["target"] == 1].sample(5),
        ], ignore_index=True)

    # --- Dataset instantiation --------------------------------------------
    if args.dataset == "Derm7pt":
        dataset_train = Derm7pt_Dataset(df_train, "train", args.dataset, transform=transforms_train)
        dataset_valid = Derm7pt_Dataset(df_valid, "valid", args.dataset, transform=transforms_val)
    elif args.dataset in ("CBD4905", "IMBD9810"):
        dataset_train = ISICCBD4905_Dataset(df_train, "train", transform=transforms_train)
        dataset_valid = ISICCBD4905_Dataset(df_valid, "valid", transform=transforms_val)
    elif args.dataset == "ISIC2017":
        dataset_train = ISIC2017_Dataset(df_train, "train", transform=transforms_train)
        dataset_valid = ISIC2017_Dataset(df_valid, "valid", transform=transforms_val)
    elif args.dataset == "ISIC2024":
        hdf5_path = os.path.join(args.data_dir, "train-image.hdf5")
        dataset_train = ISIC2024_Dataset(df_train, "train", transform=transforms_train, hdf5_path=hdf5_path)
        dataset_valid = ISIC2024_Dataset(df_valid, "valid", transform=transforms_val,   hdf5_path=hdf5_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # --- DataLoaders ------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset_train),
        num_workers=args.num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Model instantiation (rho wired into SkewMod at construction) -----

    model = HiPerViT(
        config=config,
        out_dim=args.out_dim,
        rho=rho,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)

    # --- Optimiser and schedulers -----------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, args.n_epochs - 1
    )
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine
    )

    # --- Checkpoint paths -------------------------------------------------
    model_file_best  = os.path.join(args.model_dir, f"{args.save_name}_best.pth")
    model_file_final = os.path.join(args.model_dir, f"{args.save_name}_final.pth")

    print(f"Train samples: {len(dataset_train)} | Valid samples: {len(dataset_valid)}")

    # --- Training loop ----------------------------------------------------
    es = EarlyStopping(patience=10)
    auc_max = 0.0
    epoch = 0
    done = False
    logs = []

    while epoch < args.n_epochs and not done:
        epoch += 1
        print(time.ctime(), f"Epoch {epoch}")

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc = val_epoch(model, valid_loader, mel_idx)

        if es(model, -auc):
            done = True

        scheduler_warmup.step()
        if epoch == 2:
            # Workaround for known off-by-one bug in warmup scheduler
            scheduler_warmup.step()

        current_lr = scheduler_warmup.get_last_lr()[0]

        log_entry = {
            "Time":       time.ctime(),
            "Epoch":      epoch,
            "lr":         f"{current_lr:.7f}",
            "train loss": f"{train_loss:.5f}",
            "valid loss": f"{val_loss:.5f}",
            "acc":        f"{acc:.4f}",
            "auc":        f"{auc:.6f}",
        }
        logs.append(log_entry)

        content = (
            f"{time.ctime()}  Epoch {epoch}, "
            f"lr: {current_lr:.7f}, "
            f"train loss: {train_loss:.5f}, "
            f"valid loss: {val_loss:.5f}, "
            f"acc: {acc:.4f}, "
            f"auc: {auc:.6f}, "
            f"EStop: [{es.status}]."
        )
        print(content)
        with open(os.path.join(args.log_dir, f"log_{args.save_name}.txt"), "a") as f:
            f.write(content + "\n")

        # Save checkpoint for every epoch
        torch.save(
            model.state_dict(),
            os.path.join(args.model_dir, f"{args.save_name}_epoch_{epoch}.pth"),
        )

        # Save best checkpoint by AUC
        if auc > auc_max:
            print(f"AUC improved ({auc_max:.6f} → {auc:.6f}).  Saving best model …")
            torch.save(model.state_dict(), model_file_best)
            auc_max = auc

    # Save final checkpoint and CSV log
    torch.save(model.state_dict(), model_file_final)
    pd.DataFrame(logs).to_csv(
        os.path.join(args.log_dir, f"log_{args.save_name}.csv"), index=False
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load the requested dataset split, compute the imbalance ratio, and
    launch :func:`run`.

    Calls the appropriate dataset module's ``get_df`` helper to obtain
    train/validation DataFrames and the melanoma class index, computes
    ``rho`` from the training split via
    :func:`calculate_imbalance_ratio_rho`, then applies image transforms
    and delegates to :func:`run`.
    """
    if args.dataset == "Derm7pt":
        df_train, df_valid, mel_idx = DatasetDerm7pt.get_derm_df(args.data_dir, args.out_dim)
    elif args.dataset == "CBD4905":
        df_train, df_valid, mel_idx = DatasetCBD4905.get_df(args.data_dir)
    elif args.dataset == "IMBD9810":
        df_train, df_valid, mel_idx = DatasetCBD4905.get_df_imbalanced_9810(args.data_dir)
    elif args.dataset == "ISIC2017":
        df_train, df_valid, mel_idx = DatasetISIC2017.get_df(args.data_dir, args.out_dim)
    elif args.dataset == "ISIC2024":
        df_train, df_valid, mel_idx = DatasetISIC2024.get_df(args.data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Dataset: {args.dataset} | train: {len(df_train)} | valid: {len(df_valid)}")

    # Compute the imbalance ratio from the training split and pass it to run()
    # so HiPerViT → Transformer → SkewMod can be conditioned on it.
    rho = calculate_imbalance_ratio_rho(df_train)

    transforms_train, transforms_val = get_transforms(args.image_size)
    run(df_train, df_valid, transforms_train, transforms_val, mel_idx, rho)


# ---------------------------------------------------------------------------
# GPU memory utility
# ---------------------------------------------------------------------------

def free_gpu_memory() -> None:
    """
    Print GPU memory usage, release the PyTorch CUDA cache, and run the
    Python garbage collector.

    Intended to be called at startup when ``--gpu-gc`` is passed, to reclaim
    any residual memory from previous processes sharing the same GPU.
    """
    print("Initial GPU memory usage:")
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU memory after cache clear:")
    print(torch.cuda.memory_summary())


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.gpu_gc:
        free_gpu_memory()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    set_seed()

    # Device setup
    device_cuda = torch.cuda.is_available()
    if device_cuda:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if int(args.CUDA_VISIBLE_DEVICES) != 0:
            torch.cuda.set_device(int(args.CUDA_VISIBLE_DEVICES))
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    device = torch.device("cuda" if device_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    print(f"[HiPerViT] Training started — loss: {criterion}")

    main()
