# %%
#imports needed for the notebook
from config import DATA_DIR
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
from pathlib import Path
from IPython.display import HTML
import plotly.graph_objects as go
from itertools import product
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader

# %%
#dictionary with the sizes, note: our convention is z,y,x thats why the smallest number is first here different to the paper
SIZES = {
    "archi1": (6, 20, 20) 
}


# %%
#method cuts out the cube, returns ...
def cut_cube(vol, z, y, x, size_zyx, pad_value=-1000): #values over the edge are padded with -1000, not sure if this even really happens
    dz, dy, dx = size_zyx
    z0, y0, x0 = z - dz // 2, y - dy // 2, x - dx // 2 #lower corner
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx #upper corner

    cube = np.full(size_zyx, pad_value, dtype=vol.dtype) #pads the cube full with -1000

    sz0, sy0, sx0 = max(z0, 0), max(y0, 0), max(x0, 0)
    sz1, sy1, sx1 = min(z1, vol.shape[0]), min(y1, vol.shape[1]), min(x1, vol.shape[2])

    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0


    cube[dz0:dz0 + (sz1 - sz0),
         dy0:dy0 + (sy1 - sy0),
         dx0:dx0 + (sx1 - sx0)] = vol[sz0:sz1, sy0:sy1, sx0:sx1] #if the cube hangs partly outside the volume shape the data still gets written in correctly
    return cube

#this method clips the HU units from -1000 to 400 as it is done in the paper. Then it normalizes it to values between 0 and 1
#TODO??? quote from paper: The mean gray-scale value was subtracted to adjust the distribution of training and testing data.
def normalize(cube):
    cube = np.clip(cube, -1000, 400).astype(np.float32)
    return (cube + 1000.0) / 1400.0

# method to cut and normalize, input is the volume of data and z,y,x of nodule candidate, output is dictionary of all 3 archi cubes with names
def cut_all(vol, z, y, x):
    return {name: normalize(cut_cube(vol, z, y, x, size)) for name, size in SIZES.items()}

# %%
#this cell is just to check and visualize the boxes


path = Path(DATA_DIR) / "masked_scans0" / "1.3.6.1.4.1.14519.5.2.1.6279.6001.241570579760883349458693655367.npz"
data = np.load(path)
vol = data["volume"]
seriesuid = path.stem
idx = pd.read_csv(path.parent / "candidates_index.csv")

pos = idx[(idx["seriesuid"] == seriesuid) & (idx["class"] == 1)]
r = pos.iloc[0]
z, y, x = int(r["voxel_z"]), int(r["voxel_y"]), int(r["voxel_x"])

cubes = cut_all(vol, z, y, x)


# %%
#loader
class LunaDataset1(Dataset):
    def __init__(self, subset_dir, neg_ratio=1, seed=0):
        self.subset_dir = Path(subset_dir)
        idx = pd.read_csv(self.subset_dir / "candidates_index.csv")
        self.pos = idx[idx["class"] == 1].reset_index(drop=True)
        self.neg = idx[idx["class"] == 0].reset_index(drop=True)
        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed)
        self.translations = list(product((-1, 0, 1), repeat=3))   # 27
        self.rotations = [0, 1, 2, 3]                             # 4
        self._cache_uid = None
        self._cache_vol = None
        self.build_samples()

    def build_samples(self):
        # each sample: (label, row_index, dz, dy, dx, k_rot)
        groups = {}   # seriesuid -> list of samples, for cache-friendly ordering

        for i in range(len(self.pos)):
            uid = self.pos.iloc[i]["seriesuid"]
            groups.setdefault(uid, [])
            for dz, dy, dx in self.translations:
                for k in self.rotations:
                    groups[uid].append((1, i, dz, dy, dx, k))

        n_neg = len(self.pos) * len(self.translations) * len(self.rotations) * self.neg_ratio
        neg_choice = self.rng.choice(len(self.neg), size=n_neg, replace=False)
        for j in neg_choice:
            uid = self.neg.iloc[j]["seriesuid"]
            groups.setdefault(uid, [])
            groups[uid].append((0, j, 0, 0, 0, 0))

        # shuffle scan order and within-scan order, keep each scan's samples contiguous
        uids = list(groups.keys())
        self.rng.shuffle(uids)
        self.samples = []
        for uid in uids:
            block = groups[uid]
            self.rng.shuffle(block)
            self.samples.extend(block)

    def _get_volume(self, uid):
        if uid != self._cache_uid:
            data = np.load(self.subset_dir / f"{uid}.npz")
            self._cache_vol = data["volume"]
            self._cache_uid = uid
        return self._cache_vol

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        label, row_i, dz, dy, dx, k = self.samples[i]
        row = self.pos.iloc[row_i] if label == 1 else self.neg.iloc[row_i]
        uid = row["seriesuid"]
        vol = self._get_volume(uid)

        z = int(row["voxel_z"]) + dz
        y = int(row["voxel_y"]) + dy
        x = int(row["voxel_x"]) + dx

        cubes = {}
        for name, size in SIZES.items():
            cube = cut_cube(vol, z, y, x, size)
            cube = np.rot90(cube, k, axes=(1, 2))      # rotate in transverse (y,x) plane
            cube = normalize(cube)                     # clip, scale to 0..1
            cubes[name] = torch.from_numpy(cube[None]) # add channel axis -> (1, Z, Y, X)

        return cubes, torch.tensor(label, dtype=torch.float32)
    

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from itertools import product


class LunaDatasetTF:
    def __init__(self, subset_dir, neg_ratio=1, seed=0):
        self.subset_dir = Path(subset_dir)

        idx = pd.read_csv(self.subset_dir / "candidates_index.csv")
        self.pos = idx[idx["class"] == 1].reset_index(drop=True)
        self.neg = idx[idx["class"] == 0].reset_index(drop=True)

        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed)

        self.translations = list(product((-1, 0, 1), repeat=3))  # 27
        self.rotations = [0, 1, 2, 3]

        self._cache_uid = None
        self._cache_vol = None

        self.build_samples()
        
    def __len__(self):
        return len(self.samples)

    # -----------------------
    # build dataset index
    # -----------------------
    def build_samples(self):
        groups = {}

        # positives
        for i in range(len(self.pos)):
            uid = self.pos.iloc[i]["seriesuid"]
            groups.setdefault(uid, [])

            for dz, dy, dx in self.translations:
                for k in self.rotations:
                    groups[uid].append((1, i, dz, dy, dx, k))

        # negatives
        n_neg = len(self.pos) * len(self.translations) * len(self.rotations) * self.neg_ratio
        neg_choice = self.rng.choice(len(self.neg), size=n_neg, replace=False)

        for j in neg_choice:
            uid = self.neg.iloc[j]["seriesuid"]
            groups.setdefault(uid, [])
            groups[uid].append((0, j, 0, 0, 0, 0))

        # shuffle
        uids = list(groups.keys())
        self.rng.shuffle(uids)

        self.samples = []
        for uid in uids:
            block = groups[uid]
            self.rng.shuffle(block)
            self.samples.extend(block)

    # -----------------------
    # cache volume loading
    # -----------------------
    def _get_volume(self, uid):
        if uid != self._cache_uid:
            data = np.load(self.subset_dir / f"{uid}.npz")
            self._cache_vol = data["volume"]
            self._cache_uid = uid
        return self._cache_vol
    
    def train_val_test_split(self, train=0.7, val=0.15, test=0.15, seed=0):
        assert abs(train + val + test - 1.0) < 1e-6

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return train_idx, val_idx, test_idx

    def set_indices(self, indices):
        self.active_indices = indices

    # -----------------------
    # single sample
    # -----------------------
    def get_sample(self, i):
        label, row_i, dz, dy, dx, k = self.samples[i]

        row = self.pos.iloc[row_i] if label == 1 else self.neg.iloc[row_i]
        uid = row["seriesuid"]

        vol = self._get_volume(uid)

        z = int(row["voxel_z"]) + dz
        y = int(row["voxel_y"]) + dy
        x = int(row["voxel_x"]) + dx

        cubes = []

        # Each size corresponds to a "channel" in Conv2D
        for name, size in SIZES.items():

            cube = cut_cube(vol, z, y, x, size)      # (Z, Y, X)
            cube = np.rot90(cube, k, axes=(1, 2))    # rotate in Y-X plane
            cube = normalize(cube)

            # IMPORTANT:
            # We DO NOT add channel axis anymore
            # We keep (Z, Y, X)

            cubes.append(cube)

        # Stack along "channel dimension"
        # Result: (Z, Y, X, channels)
        cubes = np.stack(cubes, axis=-1)

        # Convert from (Z, Y, X, C) → (Y, X, Z*C?) NO
        # Instead we want (Y, X, Z*C) OR better (Y, X, C_Z)

        # BEST FOR YOUR MODEL (20, 20, 6):
        # assume single scale and Z=6

        cubes = cubes[:, :, :, 0]  # remove scale axis if only 1 entry in SIZES
        cubes = np.transpose(cubes, (1, 2, 0))  # (Y, X, Z)

        return cubes.astype(np.float32), np.float32(label)
    
def make_tf_dataset(luna, batch_size=4, shuffle=True):

    def gen():
        for i in range(len(luna)):
            yield luna.get_sample(i)

    shape = luna.get_sample(0)[0].shape

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )

    if shuffle:
        ds = ds.shuffle(5000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
# %%
#ds = LunaDataset(Path(DATA_DIR) / "masked_scans0", neg_ratio=1)
#loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

#cubes, labels = next(iter(loader))
luna_dataset = LunaDatasetTF(Path(DATA_DIR) / "masked_scans0", neg_ratio=1)
ds = make_tf_dataset(luna_dataset)
batch = next(iter(ds))
# %%
def compute_means(dataset, n_augmentations = 108):
    """Mean voxel value per architecture over the dataset's current epoch
    composition (positives + drawn negatives), matching the paper's
    full-training-set centering. Augmentation is simplified: translation by +-1 doesnt change a cubes sum by much"""
    
    sums = {name: 0.0 for name in SIZES}
    counts = {name: 0 for name in SIZES}


    seen = set()
    for label, row_i, *_ in dataset.samples:
        src = dataset.pos if label == 1 else dataset.neg
        key = (label, row_i)
        if key in seen:
            continue
        seen.add(key)

        row = src.iloc[row_i]
        vol = dataset._get_volume(row["seriesuid"])
        z, y, x = int(row["voxel_z"]), int(row["voxel_y"]), int(row["voxel_x"])
        for name, size in SIZES.items():
            cube = normalize(cut_cube(vol, z, y, x, size))
            if label == 1:
                sums[name] += cube.sum() * n_augmentations
                counts[name] += cube.size * n_augmentations
            else:
                sums[name] += cube.sum()
                counts[name] += cube.size


    return {name: sums[name] / counts[name] for name in SIZES}


# %%
luna = LunaDatasetTF(Path(DATA_DIR) / "masked_scans0", neg_ratio=1)

train_idx, val_idx, test_idx = luna.train_val_test_split()

luna.set_indices(train_idx)
train_ds = make_tf_dataset(luna)

luna.set_indices(val_idx)
val_ds = make_tf_dataset(luna)

luna.set_indices(test_idx)
test_ds = make_tf_dataset(luna, shuffle=False)