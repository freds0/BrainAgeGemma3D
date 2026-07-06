#!/usr/bin/env python3
"""
Open-BHB Brain-Age Dataset
==========================
torch Dataset for the pre-processed Open-BHB volumes used for brain-age
regression.

Directory layout (read-only):
    <root>/
      train.tsv                 participant_id, age, sex, site, ...
      test.tsv
      train/quasiraw_3d/<id>_quasiraw_3d.npy   (182,218,182) float32
      train/vbm_3d/<id>_vbm_3d.npy             (121,145,121) float32
      test/...

The volumes are already skull-stripped / registered numpy arrays, so unlike
``load_nifti_volume`` in braingemma3d_architecture.py there is no NIfTI
orientation handling here: we only robust-normalize and resize.

The regression target is ``age`` (years). Optionally z-scored with
``age_mean``/``age_std`` computed on the train split for stable MSE training;
de-normalize predictions before reporting MAE in years.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _robust_normalize(vol: np.ndarray) -> np.ndarray:
    """Scale to [0,1] using the 1st-99th percentile range (per volume)."""
    vmin, vmax = np.percentile(vol, 1), np.percentile(vol, 99)
    if vmax > vmin:
        return np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.zeros_like(vol)


class OpenBHBAgeDataset(Dataset):
    """Maps a pre-processed Open-BHB volume to its subject age."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        modality: str = "quasiraw_3d",
        target_size=(64, 128, 128),
        age_mean: float = None,
        age_std: float = None,
    ):
        """
        Args:
            root: Dataset root (contains train.tsv/test.tsv and split dirs).
            split: "train" or "test".
            modality: "quasiraw_3d" (default) or "vbm_3d".
            target_size: (D,H,W) the volume is trilinearly resized to.
            age_mean, age_std: if both given, targets are z-scored.
        """
        self.dir = Path(root) / split / modality
        self.suffix = f"_{modality}.npy"
        self.target_size = tuple(target_size)
        self.age_mean = age_mean
        self.age_std = age_std

        tsv = Path(root) / f"{split}.tsv"
        if not tsv.exists():
            raise FileNotFoundError(f"Label file not found: {tsv}")
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Volume directory not found: {self.dir}")

        df = pd.read_csv(tsv, sep="\t", dtype={"participant_id": str})
        # Keep only subjects whose .npy actually exists on disk.
        have = df["participant_id"].map(
            lambda p: (self.dir / f"{p}{self.suffix}").exists()
        )
        df = df[have].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No volumes matched labels in {self.dir}")

        self.ids = df["participant_id"].tolist()
        self.ages = df["age"].astype("float32").tolist()

        n_missing = int((~have).sum())
        if n_missing:
            print(f"[OpenBHBAgeDataset] {split}/{modality}: "
                  f"{len(self.ids)} volumes ({n_missing} labels had no .npy)")

    @staticmethod
    def compute_age_stats(root: str, split: str = "train"):
        """Return (mean, std) of age over a split's label file."""
        df = pd.read_csv(Path(root) / f"{split}.tsv", sep="\t")
        ages = df["age"].astype("float32").to_numpy()
        return float(ages.mean()), float(ages.std())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        path = self.dir / f"{self.ids[i]}{self.suffix}"
        vol = np.load(path).astype(np.float32)              # (D,H,W)
        vol = _robust_normalize(vol)

        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        vol_t = F.interpolate(
            vol_t, size=self.target_size, mode="trilinear", align_corners=False
        )
        vol_t = vol_t.squeeze(0).contiguous()                # (1,D,H,W)

        age = float(self.ages[i])
        if self.age_mean is not None and self.age_std:
            age = (age - self.age_mean) / self.age_std
        return vol_t, torch.tensor(age, dtype=torch.float32)


if __name__ == "__main__":
    # Quick self-test against the real dataset.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/fred/FRED5TB/Einstein/Open_BHB_processado")
    ap.add_argument("--modality", default="quasiraw_3d")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    mean, std = OpenBHBAgeDataset.compute_age_stats(args.root, "train")
    ds = OpenBHBAgeDataset(args.root, args.split, args.modality,
                           age_mean=mean, age_std=std)
    vol, age = ds[0]
    print(f"dataset size={len(ds)} | age_mean={mean:.2f} age_std={std:.2f}")
    print(f"sample vol={tuple(vol.shape)} dtype={vol.dtype} "
          f"range=[{vol.min():.3f},{vol.max():.3f}] | age_norm={age.item():.3f}")
