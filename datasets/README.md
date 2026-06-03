
## Dataset Modules

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

```
dataset/
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
