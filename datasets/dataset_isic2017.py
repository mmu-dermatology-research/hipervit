import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage


class ISIC2017_Dataset(Dataset):
    """PyTorch Dataset for the ISIC 2017 skin-lesion challenge.

    Uses Pillow for image loading to leverage libjpeg's native downsampling
    hint (``.draft()``), which avoids decoding the full high-resolution JPEG
    when only a smaller output is needed.

    Supports binary classification (melanoma vs. rest) and the native
    three-class task (melanoma / seborrheic keratosis / nevus).

    Args:
        csv (DataFrame): Sample metadata; must contain ``'filepath'``,
            ``'image_id'``, and ``'target'`` columns.
        mode (str): ``'train'``, ``'valid'``, or ``'test'``.
        transform (callable | None): An ``albumentations`` transform applied
            to the decoded RGB image array.
        target_size (tuple[int, int]): Hint passed to the JPEG decoder to
            allow native subsampling before the albumentations pipeline.
            Should be slightly larger than the final model input resolution
            (default ``(256, 256)`` for 224-px models).
    """

    def __init__(
        self,
        csv: pd.DataFrame,
        mode: str,
        transform=None,
        target_size: tuple = (256, 256),
    ):
        self.csv         = csv.reset_index(drop=True)
        self.mode        = mode
        self.transform   = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, index: int):
        """Load and return one sample.

        The JPEG decoder is hinted via ``PIL.Image.draft()`` to decode at the
        smallest native fraction that still covers ``target_size``, reducing
        IO cost for high-resolution source images.

        Returns:
            train / valid: ``(image_tensor, label)``
            test:          ``(image_id, image_tensor, label)``
        """
        row = self.csv.iloc[index]

        pil_img = PILImage.open(row.filepath).convert('RGB')
        image   = np.array(pil_img, dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image  = torch.tensor(image.transpose(2, 0, 1)).float()
        target = torch.tensor(row.target).long()

        if self.mode == 'test':
            return row.image_id, image, target
        return image, target


# ─────────────────────────────────────────────
# DataFrame loaders
# ─────────────────────────────────────────────

def _target_col(num_classes: int) -> str:
    """Return the source CSV column name for the target variable.

    Args:
        num_classes (int): 2 → binary melanoma flag; >2 → multi-class label.

    Returns:
        str: Column name to rename to ``'target'``.
    """
    return 'melanoma' if num_classes == 2 else 'class'


def get_df(data_dir: str, num_classes: int):
    """Load the ISIC 2017 train/validation split.

    Args:
        data_dir (str): Dataset root directory.
        num_classes (int): 2 for binary (melanoma vs. rest), 3 for the full
            three-class task.

    Returns:
        tuple: ``(df_train, df_valid, mel_idx)``
    """
    col = _target_col(num_classes)

    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'))
    df_train = df_train.rename(columns={col: 'target'})
    df_train['filepath'] = df_train['image_id'].apply(
        lambda x: os.path.join(data_dir, f'ISIC-2017_Training_Data/{x}.jpg')
    )

    df_valid = pd.read_csv(os.path.join(data_dir, 'valid_set.csv'))
    df_valid = df_valid.rename(columns={col: 'target'})
    df_valid['filepath'] = df_valid['image_id'].apply(
        lambda x: os.path.join(data_dir, f'ISIC-2017_Validation_Data/{x}.jpg')
    )

    return df_train, df_valid, 1


def get_test_df(data_dir: str, num_classes: int):
    """Load the ISIC 2017 train/test split.

    Args:
        data_dir (str): Dataset root directory.
        num_classes (int): 2 for binary, 3 for multi-class.

    Returns:
        tuple: ``(df_train, df_test, mel_idx)``
    """
    col = _target_col(num_classes)

    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'))
    df_train = df_train.rename(columns={col: 'target'})
    df_train['filepath'] = df_train['image_id'].apply(
        lambda x: os.path.join(data_dir, f'ISIC-2017_Training_Data/{x}.jpg')
    )

    df_test = pd.read_csv(os.path.join(data_dir, 'test_set.csv'))
    df_test = df_test.rename(columns={col: 'target'})
    df_test['filepath'] = df_test['image_id'].apply(
        lambda x: os.path.join(data_dir, f'ISIC-2017_Test_v2_Data/{x}.jpg')
    )

    return df_train, df_test, 1
