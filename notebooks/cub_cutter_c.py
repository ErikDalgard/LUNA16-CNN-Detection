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
        shape = (dz, dy, dx, 1) if self.layout == "3d" else (dy, dx, dz)
        return tf.TensorSpec(shape=shape, dtype=tf.float32)

    def output_signature(self):
        if self.arch is None:
            feat = tuple(self._spec(name) for name in SIZES)
        else:
            feat = self._spec(self.arch)
        return feat, tf.TensorSpec(shape=(1,), dtype=tf.float32)

    def get_sample(self, i):
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



#NEW LOADING SEQUENCE: Disk → load 64 patches → feed to model → discard → load next 64 patches → .... TALK ABOUT THIS IN PROJECT
class LunaSequence(keras.utils.Sequence):
    """
    Lazy-loading Keras Sequence for LUNA16.
    Loads one batch at a time — no full-dataset materialization needed.
        """
    def __init__(self, luna_dataset, indices, batch_size=64, is_ae=False, shuffle=True):
        super().__init__(workers=4, use_multiprocessing=False)
        self.luna        = luna_dataset
        self.indices     = np.array(indices, dtype=np.int64)
        self.batch_size  = batch_size
        self.is_ae       = is_ae
        self.shuffle     = shuffle
        self._feat_shape = tuple(luna_dataset.output_signature()[0].shape)
        self._labels     = None   # populated lazily for class-weight calculation


    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        batch = self.indices[start : start + self.batch_size]

        X = np.empty((len(batch), *self._feat_shape), dtype=np.float32)
        y = np.empty((len(batch), 1),                 dtype=np.float32)
        for k, i in enumerate(batch):
            X[k], y[k] = self.luna.get_sample(int(i))

        # For autoencoders the target is the input itself
        return (X, X) if self.is_ae else (X, y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)   # randomise order each epoch

    @property
    def all_labels(self):
        """
        Reads labels straight from the sample metadata (no volume I/O).
        Used only for computing class weights in train_network.
        """
        if self._labels is None:
            self._labels = np.array(
                [self.luna.samples[int(i)][0] for i in self.indices],
                dtype=np.float32
            )
        return self._labels

def make_sequences(arch, layout, PROJECT_DIR, subset="masked_scans_0-2",
                   val_fraction=0.2, neg_ratio=1, seed=0,
                   batch_size=64, is_ae=False):

    subset_dir = Path(PROJECT_DIR) / f"data/{subset}"
    luna = LunaDataset(subset_dir, neg_ratio=neg_ratio, seed=seed, arch=arch, layout=layout)

    sample_uids = np.array([
        (luna.pos if label == 1 else luna.neg).iloc[row_i]["seriesuid"]
        for label, row_i, *_ in luna.samples
    ])
    unique  = np.unique(sample_uids)
    n_val   = max(1, int(round(val_fraction * len(unique))))
    val_uids = set(np.random.default_rng(seed).choice(unique, size=n_val, replace=False))
    is_val  = np.array([u in val_uids for u in sample_uids])

    train_seq = LunaSequence(luna, np.where(~is_val)[0], batch_size=batch_size, is_ae=is_ae, shuffle=True)
    val_seq   = LunaSequence(luna, np.where( is_val)[0], batch_size=batch_size, is_ae=is_ae, shuffle=False)

    print(f"{arch} {layout} — train batches: {len(train_seq)}, val batches: {len(val_seq)}")
    return train_seq, val_seq
