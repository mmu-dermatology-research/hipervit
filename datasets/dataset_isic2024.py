import os
import io
import numpy as np
import pandas as pd
import cv2
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class ISIC2024_Dataset(Dataset):
    """PyTorch Dataset for the ISIC 2024 skin-lesion challenge.

    Images can be loaded either from an HDF5 archive (preferred for speed)
    or directly from JPEG files on disk.  An optional resolution-degradation
    pipeline can be enabled via ``degradation_size`` to simulate low-quality
    inputs.

    Args:
        csv (DataFrame): Sample metadata; must contain ``'isic_id'``,
            ``'filepath'`` (used if ``hdf5_path`` is None), and ``'target'``
            columns.
        mode (str): ``'train'``, ``'valid'``, or ``'test'``.
        transform (callable | None): An ``albumentations`` transform applied
            after image loading and optional degradation.
        hdf5_path (str | None): Path to the ``train-image.hdf5`` archive.
            If ``None``, images are read from ``row.filepath``.
    """

    def __init__(
        self,
        csv: pd.DataFrame,
        mode: str,
        transform=None,
        hdf5_path: str = None,
    ):
        self.csv        = csv.reset_index(drop=True)
        self.mode       = mode
        self.transform  = transform
        self.hdf5_path  = hdf5_path
        self.hdf5       = None

        # Set to an integer (e.g. 128, 64) to enable downsample→upsample degradation.
        self.degradation_size: int | None = None

        if self.hdf5_path is not None:
            self.hdf5 = h5py.File(self.hdf5_path, 'r')

    def __len__(self) -> int:
        return len(self.csv)

    def __del__(self):
        if self.hdf5 is not None:
            self.hdf5.close()

    def __getitem__(self, index: int):
        """Load and return one sample.

        When ``degradation_size`` is set, the image is first downsampled to
        ``degradation_size × degradation_size`` and then upsampled back to
        224 × 224 using bilinear interpolation, simulating a low-resolution
        acquisition.

        Returns:
            train / valid: ``(image_tensor, label)``
            test:          ``(isic_id, image_tensor, label)``
        """
        row = self.csv.iloc[index]

        if self.hdf5 is not None:
            image = cv2.cvtColor(
                np.uint8(Image.open(io.BytesIO(self.hdf5[row.isic_id][()]))),
                cv2.COLOR_BGR2RGB,
            )
        else:
            image = cv2.cvtColor(cv2.imread(row.filepath), cv2.COLOR_BGR2RGB)

        if self.degradation_size is not None:
            image = cv2.resize(
                image, (self.degradation_size, self.degradation_size),
                interpolation=cv2.INTER_LINEAR,
            )
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

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
    """Load the ISIC 2024 train/validation split.

    Args:
        data_dir (str): Dataset root directory containing ``train_set.csv``
            and ``valid_set.csv``.

    Returns:
        tuple: ``(df_train, df_valid, mel_idx)``
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'), low_memory=False)
    df_valid = pd.read_csv(os.path.join(data_dir, 'valid_set.csv'))
    return df_train, df_valid, 1


def get_test_df(data_dir: str):
    """Load the ISIC 2024 train/test split.

    Args:
        data_dir (str): Dataset root directory containing ``train_set.csv``
            and ``test_set.csv``.

    Returns:
        tuple: ``(df_train, df_test, mel_idx)``
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'), low_memory=False)
    df_test  = pd.read_csv(os.path.join(data_dir, 'test_set.csv'))
    return df_train, df_test, 1
