import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class Derm7pt_Dataset(Dataset):
    """PyTorch Dataset for the Derm7pt dermoscopy collection.

    Supports both the dermoscopy (``'derm'``) and clinical photography
    (``'clinic'``) image modalities controlled by the ``dataset`` argument.

    Args:
        csv (DataFrame): Sample metadata; must contain ``'filepath'`` and
            ``'target'`` columns.
        mode (str): ``'train'``, ``'valid'``, or ``'test'``.
        dataset (str): ``'Derm7pt'`` (dermoscopy) or ``'Derm7ptClinic'``
            (clinical photography).  Controls which image-ID column is
            returned in test mode.
        transform (callable | None): An ``albumentations`` transform applied
            to the raw RGB image array.
    """

    def __init__(self, csv: pd.DataFrame, mode: str, dataset: str, transform=None):
        self.csv       = csv.reset_index(drop=True)
        self.mode      = mode
        self.dataset   = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int):
        """Load and return one sample.

        Returns:
            train / valid: ``(image_tensor, label)``
            test:          ``(image_id, image_tensor, label)``
                           where ``image_id`` is ``row.derm`` for Derm7pt
                           and ``row.clinic`` for Derm7ptClinic.
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
            image_id = row.derm if self.dataset == 'Derm7pt' else row.clinic
            return image_id, image, target
        return image, target


# ─────────────────────────────────────────────
# DataFrame loaders
# ─────────────────────────────────────────────

def _build_df(data_dir: str, split_csv: str, modality: str, num_classes: int) -> pd.DataFrame:
    """Shared helper: load a CSV, assign targets, and build filepath column.

    Args:
        data_dir (str): Dataset root directory.
        split_csv (str): CSV filename relative to ``data_dir``.
        modality (str): ``'derm'`` or ``'clinic'`` — the image column name.
        num_classes (int): 2 for binary (melanoma vs. rest), >2 for the
            native multi-class labels.

    Returns:
        DataFrame with ``'target'`` and ``'filepath'`` columns added.
    """
    df = pd.read_csv(os.path.join(data_dir, split_csv))
    if num_classes == 2:
        df['target'] = df['diagnosis'].apply(lambda x: 1 if 'melanoma' in x else 0)
    else:
        df = df.rename(columns={'class': 'target'})
    df['filepath'] = df[modality].apply(lambda x: os.path.join(data_dir, 'images', x))
    return df


def get_derm_df(data_dir: str, num_classes: int):
    """Load the Derm7pt dermoscopy train/validation split.

    Args:
        data_dir (str): Dataset root directory.
        num_classes (int): 2 for binary, >2 for multi-class.

    Returns:
        tuple: ``(df_train, df_valid, mel_idx)``
    """
    df_train = _build_df(data_dir, 'train_set.csv', 'derm', num_classes)
    df_valid = _build_df(data_dir, 'valid_set.csv', 'derm', num_classes)
    return df_train, df_valid, 1


def get_derm_test_df(data_dir: str, num_classes: int):
    """Load the Derm7pt dermoscopy train/test split.

    Args:
        data_dir (str): Dataset root directory.
        num_classes (int): 2 for binary, >2 for multi-class.

    Returns:
        tuple: ``(df_train, df_test, mel_idx)``
    """
    df_train = _build_df(data_dir, 'train_set.csv', 'derm', num_classes)
    df_test  = _build_df(data_dir, 'test_set.csv',  'derm', num_classes)
    return df_train, df_test, 1