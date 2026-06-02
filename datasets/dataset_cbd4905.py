import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class ISICCBD4905_Dataset(Dataset):
    """PyTorch Dataset for the ISIC CBD-4905 skin-lesion collection.

    Loads images from disk using paths stored in the ``'filepath'`` column of
    the provided DataFrame.

    Args:
        csv (DataFrame): Sample metadata; must contain ``'filepath'`` and
            ``'target'`` columns.
        mode (str): ``'train'``, ``'valid'``, or ``'test'``.  In ``'test'``
            mode the sample's ``isic_id`` is returned alongside the data and
            label so predictions can be traced back to individual images.
        transform (callable | None): An ``albumentations`` transform applied
            to the raw RGB image array.
    """

    def __init__(self, csv: pd.DataFrame, mode: str, transform=None):
        self.csv       = csv.reset_index(drop=True)
        self.mode      = mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int):
        """Load and return one sample.

        Returns:
            train / valid: ``(image_tensor, label)``
            test:          ``(isic_id, image_tensor, label)``
        """
        row   = self.csv.iloc[index]
        image = cv2.cvtColor(cv2.imread(row.filepath), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image  = torch.tensor(image.transpose(2, 0, 1)).float()
        target = torch.tensor(row.target).long()

        if self.mode == 'test':
            return row.isic_id, image, target
        return image, target


# ─────────────────────────────────────────────
# DataFrame loaders
# ─────────────────────────────────────────────

def get_df(data_dir: str):
    """Load the balanced CBD-4905 train/validation split.

    Args:
        data_dir (str): Root directory containing the CSV files and image
            sub-directories.

    Returns:
        tuple: ``(df_train, df_valid, mel_idx)``
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'balanced-4905-train-complete-metadata.csv'))
    df_train = df_train.rename(columns={'class': 'target'})
    df_train['filepath'] = df_train.apply(
        lambda r: os.path.join(
            data_dir,
            'train_balanced_224x224/train/mel' if r['target'] == 1
            else 'train_balanced_224x224/train/oth',
            r['image'],
        ),
        axis=1,
    )

    df_valid = pd.read_csv(os.path.join(data_dir, 'balanced-4905-val-complete-metadata.csv'))
    df_valid = df_valid.rename(columns={'class': 'target'})
    df_valid['filepath'] = df_valid.apply(
        lambda r: os.path.join(
            data_dir,
            'train_balanced_224x224/val/mel' if r['target'] == 1
            else 'train_balanced_224x224/val/oth',
            r['image'],
        ),
        axis=1,
    )

    return df_train, df_valid, 1  # mel_idx = 1


def get_df_imbalanced_9810(data_dir: str):
    """Load the 98:10 imbalanced ISIC train/validation split.

    Args:
        data_dir (str): Root directory containing the CSV files.

    Returns:
        tuple: ``(df_train, df_valid, mel_idx)``
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'isic_imbalanced_9810_train_set.csv'))
    df_train = df_train.rename(columns={'class': 'target'})

    df_valid = pd.read_csv(os.path.join(data_dir, 'isic_imbalanced_9810_valid_set.csv'))
    df_valid = df_valid.rename(columns={'class': 'target'})

    return df_train, df_valid, 1
