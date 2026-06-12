"""Cube extraction and tf.data input pipeline for the LUNA16 multi-scale FP-reduction project.

Importable, TensorFlow-only, no side effects at import time.

Example:
    from cub_cutter import LunaDataset, make_tf_dataset, compute_means

    # one architecture, Conv3D input -> batched (B, Z, Y, X, 1)
    ds   = LunaDataset(subset_dir, arch="archi1", layout="3d", neg_ratio=1)
    tfds = make_tf_dataset(ds, batch_size=64)
    model.fit(tfds, validation_data=val_tfds, epochs=50)

    # same data for the 2D models -> batched (B, Y, X, Z) == (B, 20, 20, 6) for archi1
    ds = LunaDataset(subset_dir, arch="archi1", layout="2d")

    # optional paper-style centering: subtract per-architecture mean after normalization
    means = compute_means(ds)
    ds    = LunaDataset(subset_dir, arch="archi1", means=means)
"""
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# z, y, x convention (smallest dimension first), one patch size per architecture.
SIZES = {
    "archi1": (6, 20, 20),
    "archi2": (10, 30, 30),
    "archi3": (26, 40, 40),
}


def cut_cube(vol, z, y, x, size_zyx, pad_value=-1000):
    """Extract a size_zyx patch centered at (z, y, x). Voxels outside the volume are padded."""
    dz, dy, dx = size_zyx
    z0, y0, x0 = z - dz // 2, y - dy // 2, x - dx // 2
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx

    cube = np.full(size_zyx, pad_value, dtype=vol.dtype)

    sz0, sy0, sx0 = max(z0, 0), max(y0, 0), max(x0, 0)
    sz1, sy1, sx1 = min(z1, vol.shape[0]), min(y1, vol.shape[1]), min(x1, vol.shape[2])

    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
    cube[dz0:dz0 + (sz1 - sz0),
         dy0:dy0 + (sy1 - sy0),
         dx0:dx0 + (sx1 - sx0)] = vol[sz0:sz1, sy0:sy1, sx0:sx1]
    return cube


def normalize(cube):
    """Clip HU to [-1000, 400] and scale to [0, 1] (float32)."""
    cube = np.clip(cube, -1000, 400).astype(np.float32)
    return (cube + 1000.0) / 1400.0


def cut_all(vol, z, y, x):
    """Normalized numpy patch per architecture: {name: (Z, Y, X) float32}."""
    return {name: normalize(cut_cube(vol, z, y, x, size)) for name, size in SIZES.items()}


class LunaDataset:
    """Builds an augmented candidate list and produces per-candidate patches.

    arch:
        None          -> features are a tuple of all three scales (multi-input model)
        "archi1/2/3"  -> features are a single patch of that scale (single-input model)
    layout:
        "3d"          -> (Z, Y, X, 1), for the Conv3D models (get_archi_*_3D)
        "2d"          -> (Y, X, Z), depth as channels, for the Conv2D models (get_archi_*_2D)
    means:
        None          -> no centering
        {name: float} -> subtract the per-architecture mean after normalization
    """

    def __init__(self, subset_dir, neg_ratio=1, seed=0, arch=None, layout="3d", means=None):
        if arch is not None and arch not in SIZES:
            raise ValueError(f"arch must be None or one of {list(SIZES)}, got {arch!r}")
        if layout not in ("2d", "3d"):
            raise ValueError(f"layout must be '2d' or '3d', got {layout!r}")

        self.subset_dir = Path(subset_dir)
        self.arch = arch
        self.layout = layout
        self.means = means
        self.neg_ratio = neg_ratio

        idx = pd.read_csv(self.subset_dir / "candidates_index.csv")
        self.pos = idx[idx["class"] == 1].reset_index(drop=True)
        self.neg = idx[idx["class"] == 0].reset_index(drop=True)

        self.rng = np.random.default_rng(seed)
        self.translations = [(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]  #create 20 new for each positive
        self.rotations = [0, 1, 2, 3]                           

        self._cache_uid = None
        self._cache_vol = None

        self.build_samples()

    def build_samples(self):
        # sample tuple: (label, row_index, dz, dy, dx, k_rot)
        groups = {}   # seriesuid -> samples, kept contiguous for volume-cache locality

        for i in range(len(self.pos)):
            uid = self.pos.iloc[i]["seriesuid"]
            groups.setdefault(uid, [])
            for dz, dy, dx in self.translations:
                for k in self.rotations:
                    groups[uid].append((1, i, dz, dy, dx, k))

        n_neg = len(self.pos) * len(self.translations) * len(self.rotations) * self.neg_ratio
        n_neg = min(n_neg, len(self.neg))   # cannot draw more unique negatives than the subset holds
        neg_choice = self.rng.choice(len(self.neg), size=n_neg, replace=False)
        for j in neg_choice:
            uid = self.neg.iloc[j]["seriesuid"]
            groups.setdefault(uid, [])
            groups[uid].append((0, int(j), 0, 0, 0, 0))

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

    def _finalize(self, cube, name):
        cube = normalize(cube)
        if self.means is not None:
            cube = cube - np.float32(self.means[name])
        if self.layout == "3d":
            cube = np.transpose(cube, (1, 2, 0))  # (Y, X, Z)
            cube = cube[..., None]                  # (Z, Y, X, 1)
            
        else:
            cube = np.transpose(cube, (1, 2, 0))    # (Y, X, Z), depth as channels
        return np.ascontiguousarray(cube, dtype=np.float32)

    def _make_cube(self, vol, z, y, x, name, k):
        cube = cut_cube(vol, z, y, x, SIZES[name])
        cube = np.rot90(cube, k, axes=(1, 2))       # rotate in the transverse (y, x) plane
        return self._finalize(cube, name)
    
    def _spec(self, name):
        dz, dy, dx = SIZES[name]

        if self.layout == "3d":
            shape = (dy, dx, dz, 1)  
        else:
            shape = (dy, dx, dz)

        return tf.TensorSpec(shape=shape, dtype=tf.float32)

    def output_signature(self):
        if self.arch is None:
            feat = tuple(self._spec(name) for name in SIZES)
        else:
            feat = self._spec(self.arch)
        return feat, tf.TensorSpec(shape=(1,), dtype=tf.float32)

    def get_sample(self, i, augment=True):
        label, row_i, dz, dy, dx, k = self.samples[i]
        row = self.pos.iloc[row_i] if label == 1 else self.neg.iloc[row_i]
        uid = row["seriesuid"]
        vol = self._get_volume(uid)

        z = int(row["voxel_z"]) + dz
        y = int(row["voxel_y"]) + dy
        x = int(row["voxel_x"]) + dx

        if self.arch is None:
            feats = tuple(self._make_cube(vol, z, y, x, name, k) for name in SIZES)
        else:
            feats = self._make_cube(vol, z, y, x, self.arch, k)
        return feats, np.array([label], dtype=np.float32)

    def __len__(self):
        return len(self.samples)


def make_tf_dataset(luna, batch_size=4, shuffle=True, shuffle_buffer=512):
    """Wrap a LunaDataset in a batched, prefetched tf.data.Dataset.

    The generator reads samples scan-contiguously so the single-volume cache stays warm;
    ds.shuffle only reorders the emitted elements, not the read order.
    """
    def gen():
        for i in range(len(luna.samples)):
            yield luna.get_sample(i)

    ds = tf.data.Dataset.from_generator(gen, output_signature=luna.output_signature())
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_means(dataset, n_augmentations=108):
    """Per-architecture mean voxel value over unique candidates, positives weighted by
    n_augmentations (27 translations x 4 rotations) to match the augmented epoch composition."""
    sums = {name: 0.0 for name in SIZES}
    counts = {name: 0 for name in SIZES}
    seen = set()

    for label, row_i, *_ in dataset.samples:
        key = (label, row_i)
        if key in seen:
            continue
        seen.add(key)

        src = dataset.pos if label == 1 else dataset.neg
        row = src.iloc[row_i]
        vol = dataset._get_volume(row["seriesuid"])
        z, y, x = int(row["voxel_z"]), int(row["voxel_y"]), int(row["voxel_x"])

        w = n_augmentations if label == 1 else 1
        for name, size in SIZES.items():
            cube = normalize(cut_cube(vol, z, y, x, size))
            sums[name] += float(cube.sum()) * w
            counts[name] += cube.size * w

    return {name: sums[name] / counts[name] for name in SIZES}


def training_validation_split(PROJECT_DIR, arch, layout):
    """Splits the training data into validation and training LundaDatasets. Uses unique UIDs per split and outputs as tf. generators"""

    data_dir = f"{PROJECT_DIR}/data/masked_scans_0-2"

    ds = LunaDataset(data_dir, arch=arch, layout=layout, neg_ratio=1)

    sample_uids = []

    for i in range(len(ds.samples)):
        label = ds.samples[i][0]
        row_i = ds.samples[i][1]

        if label == 1:
            row = ds.pos.iloc[row_i]
        else:
            row = ds.neg.iloc[row_i]

        uid = row["seriesuid"]
        sample_uids.append(uid)

    sample_uids = np.array(sample_uids)

    # -----------------------------
    # 3. Split by unique UIDs (patient-level split)
    # -----------------------------
    unique_uids = np.unique(sample_uids)

    rng = np.random.default_rng(0)
    rng.shuffle(unique_uids)

    val_fraction = 0.2
    n_val = int(len(unique_uids) * val_fraction)

    val_uids = set()
    i = 0
    while i < n_val:
        val_uids.add(unique_uids[i])
        i = i + 1

    # -----------------------------
    # 4. Build index masks
    # -----------------------------
    train_indices = []
    val_indices = []

    i = 0
    while i < len(sample_uids):
        uid = sample_uids[i]

        if uid in val_uids:
            val_indices.append(i)
        else:
            train_indices.append(i)

        i = i + 1

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # -----------------------------
    # 5. tf.data generators
    # -----------------------------
    def train_gen():
        i = 0
        while i < len(train_indices):
            idx = train_indices[i]
            x, y = ds.get_sample(idx, augment=True)
            yield x, y
            i = i + 1

    def val_gen():
        i = 0
        while i < len(val_indices):
            idx = val_indices[i]
            x, y = ds.get_sample(idx, augment=False)
            yield x, y
            i = i + 1

    # -----------------------------
    # 6. Output signature
    # -----------------------------
    output_signature = ds.output_signature()

    # -----------------------------
    # 7. Build tf.data datasets
    # -----------------------------
    train_tfds = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=output_signature
    )

    val_tfds = tf.data.Dataset.from_generator(
        val_gen,
        output_signature=output_signature
    )

    train_tfds = train_tfds.batch(64).prefetch(tf.data.AUTOTUNE)
    val_tfds = val_tfds.batch(64).prefetch(tf.data.AUTOTUNE)

    return train_tfds, val_tfds