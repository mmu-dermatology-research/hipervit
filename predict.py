"""
predict.py
----------
Inference / evaluation entry-point for HiPerViT.

Loads a saved checkpoint, runs the model over the test split of a supported
dermatology dataset with optional test-time augmentation (TTA), and reports
a comprehensive set of classification metrics.

Supported datasets
------------------
- ISIC2017
- ISIC2018
- ISIC2024
- Derm7pt  (dermoscopy split)

Metrics reported
----------------
Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion matrix,
per-class accuracy, mean accuracy, and a full classification report.

"""

import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_trans, get_transforms

import datasets.dataset_isic2024  as DatasetISIC2024
import datasets.dataset_isic2017  as DatasetISIC2017
import datasets.dataset_isic2018  as DatasetISIC2018
import datasets.dataset_derm7pt   as DatasetDerm7pt

from datasets.dataset_isic2024 import ISIC2024_Dataset
from datasets.dataset_isic2017 import ISIC2017_Dataset
from datasets.dataset_isic2018 import ISIC2018_Dataset
from datasets.dataset_derm7pt  import Derm7pt_Dataset

from models.hipervit import HiPerViT


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a trained HiPerViT checkpoint."
    )
    parser.add_argument("--kernel-type", type=str, required=True,
                        help="Checkpoint base name (matches the --save-name used during training).")
    parser.add_argument("--data-dir",    type=str, default="/raid/",
                        help="Root directory containing the dataset.")
    parser.add_argument("--image-size",  type=int, required=True,
                        help="Input image resolution (square).")
    parser.add_argument("--enet-type",   type=str, required=True,
                        help="EfficientNet variant key (e.g. 'efficientnet_b0').")
    parser.add_argument("--pretrained",  action="store_true",
                        help="Initialise EfficientNet backbone with ImageNet weights.")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--out-dim",     type=int, default=2, help="Number of output classes.")
    parser.add_argument("--DEBUG",       action="store_true",
                        help="Debug mode: evaluate on a small subset (25 neg + 5 pos).")
    parser.add_argument("--model-dir",   type=str, default="./weights",
                        help="Directory containing saved model checkpoints.")
    parser.add_argument("--sub-dir",     type=str, default="./subs",
                        help="Directory for saving prediction result text files.")
    parser.add_argument("--eval",        type=str, default="best",
                        choices=["best", "final"],
                        help="Which checkpoint to load: 'best' (highest AUC) or 'final'.")
    parser.add_argument("--n-test",      type=int, default=8,
                        help="Number of test-time augmentation (TTA) rounds.")
    parser.add_argument("--dataset",     type=str, required=True,
                        choices=["ISIC2017", "ISIC2018", "ISIC2024", "Derm7pt"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--config",      type=str, default="./config.yaml",
                        help="Path to the YAML architecture configuration file.")

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
# Model loading helper
# ---------------------------------------------------------------------------

def load_model(model_file: str) -> nn.Module:
    """
    Instantiate HiPerViT and load weights from a checkpoint file.

    Handles both single-GPU checkpoints (keys as-is) and multi-GPU /
    DataParallel checkpoints (keys prefixed with ``"module."``).

    Args:
        model_file: Path to the ``.pth`` checkpoint file.

    Returns:
        HiPerViT model in ``eval()`` mode, moved to the global ``device``.
    """
    model = HiPerViT(
        config=config,
        out_dim=args.out_dim
    )
    model = model.to(device)

    try:
        # Single-GPU checkpoint
        model.load_state_dict(
            torch.load(model_file, map_location=device), strict=True
        )
    except RuntimeError:
        # DataParallel checkpoint — strip the "module." prefix
        state_dict = torch.load(model_file, map_location=device)
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load dataset, run inference with TTA across all loaded models, compute
    and print evaluation metrics, and save results to ``--sub-dir``.

    Metrics computed
    ----------------
    - Accuracy, Precision, Recall, F1-score (binary, melanoma class)
    - ROC-AUC
    - Confusion matrix (TN, FP, FN, TP)
    - Per-class accuracy and mean accuracy
    - Full sklearn ``classification_report``
    """
    # --- Load dataset split -----------------------------------------------
    if args.dataset == "Derm7pt":
        df_train, df_test, mel_idx = DatasetDerm7pt.get_derm_test_df(args.data_dir, args.out_dim)
    elif args.dataset == "ISIC2017":
        df_train, df_test, mel_idx = DatasetISIC2017.get_test_df(args.data_dir, args.out_dim)
    elif args.dataset == "ISIC2018":
        df_train, df_test, mel_idx = DatasetISIC2018.get_test_df(args.data_dir)
    elif args.dataset == "ISIC2024":
        df_train, df_test, mel_idx = DatasetISIC2024.get_test_df(args.data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Debug mode: evaluate on a small balanced subset
    if args.DEBUG:
        df_test = pd.concat([
            df_test[df_test["target"] == 0].sample(25),
            df_test[df_test["target"] == 1].sample(5),
        ], ignore_index=True)

    # --- Transforms and dataset -------------------------------------------
    _, transforms_val = get_transforms(args.image_size)

    if args.dataset == "Derm7pt":
        dataset_test = Derm7pt_Dataset(df_test, "test", args.dataset, transform=transforms_val)
    elif args.dataset == "ISIC2017":
        dataset_test = ISIC2017_Dataset(df_test, "test", transform=transforms_val)
    elif args.dataset == "ISIC2018":
        dataset_test = ISIC2018_Dataset(df_test, "test", transform=transforms_val)
    elif args.dataset == "ISIC2024":
        dataset_test = ISIC2024_Dataset(
            df_test, "test",
            transform=transforms_val,
            hdf5_path=os.path.join(args.data_dir, "train-image.hdf5"),
        )

    test_loader = DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # --- Resolve checkpoint path ------------------------------------------
    if args.eval == "best":
        model_file = os.path.join(args.model_dir, f"{args.kernel_type}_best.pth")
    else:  # "final"
        model_file = os.path.join(args.model_dir, f"{args.kernel_type}_final.pth")

    print(f"Loading checkpoint: {model_file}")

    # --- Load model -------------------------------------------------------
    models = [load_model(model_file)]

    # --- Inference with TTA -----------------------------------------------
    PROBS, TARGETS = [], []

    with torch.no_grad():
        for sample_id, data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for model in models:
                for I in range(args.n_test):
                    probs += model(get_trans(data, I)).softmax(1)

            probs /= args.n_test
            probs /= len(models)

            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

    PROBS   = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # --- Metrics ----------------------------------------------------------
    y_pred = PROBS.argmax(1)
    y_true = TARGETS

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    roc_auc   = roc_auc_score((y_true == mel_idx).astype(float), PROBS[:, mel_idx])
    cm        = confusion_matrix(y_true, y_pred).ravel()  # (TN, FP, FN, TP)

    conf_matrix   = confusion_matrix(y_true, y_pred)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    mean_acc      = float(np.mean(per_class_acc))
    class_report  = classification_report(y_true, y_pred, target_names=["NON-MEL", "MEL"])

    # --- Console output ---------------------------------------------------
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Confusion matrix (TN, FP, FN, TP): {cm}")
    print(f"Per-class accuracy: {per_class_acc}")
    print(f"Mean accuracy:      {mean_acc:.4f}")
    print(f"\nClassification report:\n{class_report}")

    # --- Save results to file ---------------------------------------------
    content = (
        f"{time.ctime()}  [{args.eval}] {args.kernel_type}  "
        f"dataset: {args.dataset}\n"
        f"  accuracy:  {acc:.4f}  precision: {precision:.4f}  "
        f"recall: {recall:.4f}  f1: {f1:.4f}  "
        f"roc_auc: {roc_auc:.4f}  (TN,FP,FN,TP): {cm}\n"
        f"  per-class accuracy: {per_class_acc}\n"
        f"  mean accuracy: {mean_acc:.4f}\n"
        f"Classification report:\n{class_report}\n"
    )

    result_path = os.path.join(args.sub_dir, f"pred_{args.kernel_type}.txt")
    with open(result_path, "a") as f:
        f.write(content + "\n")

    print(f"Results saved to: {result_path}")


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.sub_dir, exist_ok=True)

    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    set_seed()

    device_cuda = torch.cuda.is_available()
    if device_cuda:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    device = torch.device("cuda" if device_cuda else "cpu")

    main()
