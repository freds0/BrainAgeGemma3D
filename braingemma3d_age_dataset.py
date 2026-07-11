#!/usr/bin/env python3
"""Open-BHB dataset utilities for BrainGemma3D age regression."""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _robust_normalize(vol: np.ndarray, invalid_policy: str = "error") -> np.ndarray:
    """Scale a volume to [0, 1], optionally repairing non-finite voxels."""
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape={vol.shape}")
    finite = np.isfinite(vol)
    if not finite.all():
        if invalid_policy != "sanitize":
            raise ValueError(
                f"Volume contains {(~finite).sum()} NaN or infinite voxels"
            )
        if finite.any():
            finite_values = vol[finite]
            replacement = float(np.median(finite_values))
            vol = np.nan_to_num(
                vol,
                nan=replacement,
                posinf=float(finite_values.max()),
                neginf=float(finite_values.min()),
            )
        else:
            vol = np.zeros_like(vol, dtype=np.float32)
    vmin, vmax = np.percentile(vol, (1, 99))
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


class OpenBHBAgeDataset(Dataset):
    """Map preprocessed Open-BHB volumes to normalized ages and metadata."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        modality: str = "quasiraw_3d",
        target_size=(32, 112, 112),
        age_mean: Optional[float] = None,
        age_std: Optional[float] = None,
        participant_ids: Optional[Iterable[str]] = None,
        return_metadata: bool = False,
        invalid_volume_policy: str = "error",
    ):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.dir = self.root / split / modality
        self.suffix = f"_{modality}.npy"
        self.target_size = tuple(target_size)
        self.age_mean = age_mean
        self.age_std = age_std
        self.return_metadata = return_metadata
        self.invalid_volume_policy = invalid_volume_policy

        if age_std is not None and (not np.isfinite(age_std) or age_std <= 0):
            raise ValueError(f"age_std must be positive and finite, got {age_std}")
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Volume directory not found: {self.dir}")

        df = self.load_metadata(root, split, modality)
        if participant_ids is not None:
            wanted = {str(value) for value in participant_ids}
            df = df[df["participant_id"].isin(wanted)]
            missing = wanted.difference(df["participant_id"])
            if missing:
                raise ValueError(
                    f"{len(missing)} requested participants have no matching volume"
                )
        if df.empty:
            raise RuntimeError(f"No volumes matched labels in {self.dir}")

        self.frame = df.reset_index(drop=True)
        self.ids = self.frame["participant_id"].tolist()
        self.ages = self.frame["age"].astype("float32").tolist()

    @staticmethod
    def load_metadata(root: str, split: str, modality: str) -> pd.DataFrame:
        """Load valid, unique labels having a corresponding volume on disk."""
        root = Path(root)
        tsv = root / f"{split}.tsv"
        volume_dir = root / split / modality
        if not tsv.exists():
            raise FileNotFoundError(f"Label file not found: {tsv}")
        if not volume_dir.is_dir():
            raise FileNotFoundError(f"Volume directory not found: {volume_dir}")

        df = pd.read_csv(tsv, sep="\t", dtype={"participant_id": str})
        required = {"participant_id", "age"}
        missing_columns = required.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing TSV columns: {sorted(missing_columns)}")
        if df["participant_id"].duplicated().any():
            raise ValueError(f"Duplicate participant IDs in {tsv}")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        if not np.isfinite(df["age"]).all():
            raise ValueError(f"Missing or non-finite ages in {tsv}")

        suffix = f"_{modality}.npy"
        have = df["participant_id"].map(
            lambda p: (volume_dir / f"{p}{suffix}").exists()
        )
        if (~have).any():
            print(
                f"[OpenBHBAgeDataset] {split}/{modality}: dropping {(~have).sum()} missing volumes"
            )
        return df[have].reset_index(drop=True)

    @staticmethod
    def inspect_volumes(root: str, split: str, modality: str, frame: pd.DataFrame):
        """Return records for unreadable, non-3D, or non-finite volumes."""
        volume_dir = Path(root) / split / modality
        suffix = f"_{modality}.npy"
        invalid = []
        total = len(frame)
        for position, participant_id in enumerate(
            frame["participant_id"].astype(str), 1
        ):
            path = volume_dir / f"{participant_id}{suffix}"
            try:
                volume = np.load(path, mmap_mode="r")
                if volume.ndim != 3:
                    raise ValueError(f"expected 3 dimensions, got {volume.shape}")
                nonfinite = int((~np.isfinite(volume)).sum())
                if nonfinite:
                    raise ValueError(f"{nonfinite} NaN/Inf voxels")
            except Exception as error:
                invalid.append(
                    {
                        "participant_id": participant_id,
                        "split": split,
                        "path": str(path),
                        "error": str(error),
                    }
                )
            if position % 250 == 0 or position == total:
                print(
                    f"[preflight] {split}/{modality}: {position}/{total} "
                    f"checked, {len(invalid)} invalid",
                    end="\r" if position < total else "\n",
                )
        return invalid

    @staticmethod
    def compute_age_stats(
        root: str,
        split: str = "train",
        modality: str = "quasiraw_3d",
        participant_ids: Optional[Iterable[str]] = None,
    ):
        """Return population mean/std over the actual selected training volumes."""
        df = OpenBHBAgeDataset.load_metadata(root, split, modality)
        if participant_ids is not None:
            wanted = {str(value) for value in participant_ids}
            df = df[df["participant_id"].isin(wanted)]
        ages = df["age"].to_numpy(dtype=np.float32)
        if not len(ages):
            raise ValueError("Cannot compute age statistics on an empty subset")
        std = float(ages.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            raise ValueError(f"Invalid age standard deviation: {std}")
        return float(ages.mean()), std

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.dir / f"{row.participant_id}{self.suffix}"
        try:
            raw = np.load(path).astype(np.float32, copy=False)
            vol = _robust_normalize(raw, self.invalid_volume_policy)
        except Exception as error:
            raise ValueError(
                f"Invalid volume for participant {row.participant_id}: {path}: {error}"
            ) from error
        vol_t = torch.from_numpy(vol)[None, None]
        vol_t = (
            F.interpolate(
                vol_t, size=self.target_size, mode="trilinear", align_corners=False
            )
            .squeeze(0)
            .contiguous()
        )

        age = float(row.age)
        target = age
        if self.age_mean is not None and self.age_std is not None:
            target = (age - self.age_mean) / self.age_std
        target_t = torch.tensor(target, dtype=torch.float32)
        if not self.return_metadata:
            return vol_t, target_t
        metadata = {
            "participant_id": str(row.participant_id),
            "age_years": age,
            "sex": str(row.get("sex", "unknown")),
            "site": str(row.get("site", "unknown")),
        }
        return vol_t, target_t, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--modality", default="quasiraw_3d")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    mean, std = OpenBHBAgeDataset.compute_age_stats(args.root, "train", args.modality)
    dataset = OpenBHBAgeDataset(
        args.root, args.split, args.modality, age_mean=mean, age_std=std
    )
    volume, age = dataset[0]
    print(f"size={len(dataset)} mean={mean:.2f} std={std:.2f}")
    print(
        f"volume={tuple(volume.shape)} range=[{volume.min():.3f}, {volume.max():.3f}]"
    )
    print(f"normalized_age={age.item():.3f}")
