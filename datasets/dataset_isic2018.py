import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class ISIC2018_Dataset(Dataset):
    """PyTorch Dataset for the ISIC 2018 skin-lesion challenge (Task 3).

    Args:
        csv (DataFrame): Sample metadata; must contain ``'filepath'``,
            ``'image'``, and ``'target'`` columns.
        mode (str): ``'train'``, ``'valid'``, or ``'test'``.  In ``'test'``
            mode the ``'image'`` identifier is returned alongside data and
            label.
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
            test:          ``(image_id, image_tensor, label)``
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
            return row.image, image, target
        return image, target


# ─────────────────────────────────────────────
# DataFrame loaders
# ─────────────────────────────────────────────

def get_test_df(data_dir: str):
    """Load the ISIC 2018 Task-3 test split.

    The training DataFrame is left empty because ISIC 2018 is used only for
    held-out evaluation in this project; ρ is computed from the train split
    of another dataset.

    Args:
        data_dir (str): Dataset root directory containing ``test_set.csv``
            and the ``ISIC2018_Task3_Test_Input/`` image folder.

    Returns:
        tuple: ``(df_train, df_test, mel_idx)``
            ``df_train`` is an empty DataFrame.
    """
    df_test = pd.read_csv(os.path.join(data_dir, 'test_set.csv'))
    df_test = df_test.rename(columns={'MEL': 'target'})
    df_test['filepath'] = df_test['image'].apply(
        lambda x: os.path.join(data_dir, f'ISIC2018_Task3_Test_Input/{x}.jpg')
    )
    return pd.DataFrame(), df_test, 1
