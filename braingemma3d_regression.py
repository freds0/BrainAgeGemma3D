#!/usr/bin/env python3
"""Reliable BrainGemma3D training for Open-BHB brain-age regression."""

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from braingemma3d_age_dataset import OpenBHBAgeDataset
from braingemma3d_architecture import MedSigLIP3D


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class ExperimentLogger:
    """Mirror scalar metrics to TensorBoard and Weights & Biases."""

    def __init__(self, args):
        self.writer = None
        self.wandb = None
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(args.tensorboard_dir or output_dir / "tensorboard")
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.writer.add_text(
                "config", json.dumps(vars(args), indent=2, default=str), 0
            )
            print(f"[tensorboard] logs -> {log_dir}")
        if args.wandb:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError(
                    "W&B logging requested; install it with: pip install wandb"
                ) from error
            self.wandb = wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                id=args.wandb_run_id,
                resume="allow" if args.wandb_run_id else None,
                mode=args.wandb_mode,
                dir=str(output_dir),
                config=vars(args),
            )
            print(f"[wandb] mode={args.wandb_mode} project={args.wandb_project}")

    def log(self, metrics, step):
        clean = {key: float(value) for key, value in metrics.items()}
        if self.writer is not None:
            for key, value in clean.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(clean, step=step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


def resolve_vision_model_dir(spec: str, cache_root: str = "models") -> str:
    if os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(cache_root, spec.replace("/", "__"))
    print(f"[weights] downloading '{spec}' -> {local_dir}")
    return snapshot_download(
        repo_id=spec,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
    )


def make_train_val_split(frame: pd.DataFrame, val_fraction: float, seed: int):
    """Stratify by site, age band and sex, collapsing only sparse strata."""
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    age_bins = pd.qcut(
        frame["age"], q=min(10, frame["age"].nunique()), duplicates="drop"
    )
    sex = frame.get("sex", pd.Series("unknown", index=frame.index)).astype(str)
    site = frame.get("site", pd.Series("unknown", index=frame.index)).astype(str)
    age = age_bins.astype(str)
    fallback = age + "|" + sex
    strata = site + "|" + age + "|" + sex
    for simpler in (site + "|" + age, site, fallback):
        counts = strata.value_counts()
        strata = strata.where(strata.map(counts) >= 2, simpler)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, "__rare__")
    if (strata == "__rare__").sum() == 1:
        strata.loc[strata == "__rare__"] = strata.mode().iloc[0]
    train_ids, val_ids = train_test_split(
        frame["participant_id"].astype(str).to_numpy(),
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    return train_ids.tolist(), val_ids.tolist()


class BrainAgeRegressor(nn.Module):
    """Inflated 3D MedSigLIP encoder with a scalar regression head."""

    def __init__(
        self,
        vision_model_dir: str,
        depth: int = 4,
        depth_stride: int = 4,
        max_depth_patches: int = 8,
        stage: str = "head",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        head_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.stage = stage
        self.encoder = MedSigLIP3D(
            model_name_or_dir=vision_model_dir,
            depth=depth,
            depth_stride=depth_stride,
            max_depth_patches=max_depth_patches,
        )
        # Text parameters are irrelevant to image regression and must never enter
        # the optimizer, including during full vision fine-tuning.
        if self.encoder.text_model is not None:
            for parameter in self.encoder.text_model.parameters():
                parameter.requires_grad = False

        for parameter in self.encoder.vision_model.parameters():
            parameter.requires_grad = False

        if stage in {"stem", "lora"}:
            for parameter in self.encoder.vision_model.patch_embedding_3d.parameters():
                parameter.requires_grad = True
        elif stage == "full":
            for parameter in self.encoder.vision_model.parameters():
                parameter.requires_grad = True
        elif stage != "head":
            raise ValueError(f"Unknown training stage: {stage}")

        if stage == "lora":
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            self.encoder.vision_model.encoder = get_peft_model(
                self.encoder.vision_model.encoder, config
            )

        vision_dim = self.encoder.vision_model.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    @property
    def encoder_requires_grad(self):
        return any(parameter.requires_grad for parameter in self.encoder.parameters())

    def forward(self, volume: torch.Tensor):
        grad_context = (
            contextlib.nullcontext() if self.encoder_requires_grad else torch.no_grad()
        )
        with grad_context:
            tokens = self.encoder.encode_image(volume)
        features = tokens.mean(dim=1)
        return self.head(features.float()).squeeze(-1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.stage in {"head", "stem"}:
            self.encoder.eval()
        return self

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def checkpoint_state(model: BrainAgeRegressor):
    if model.stage == "full":
        return model.state_dict()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value for name, value in model.state_dict().items() if name in trainable
    }


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, best_val_mae, args, age_stats
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": checkpoint_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "age_mean": age_stats[0],
        "age_std": age_stats[1],
        "args": vars(args),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)
    print(f"[save] {len(payload['model_state'])} tensors -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, restore_rng=False):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state", payload.get("state"))
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng:
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    print(f"[load] {path} | missing base keys={len(missing)}")
    return payload


def regression_metrics(predictions, targets):
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    error = pred - true
    denominator = np.sum((true - true.mean()) ** 2)
    pearson = float(np.corrcoef(pred, true)[0, 1]) if len(true) > 1 else float("nan")
    spearman = float(pd.Series(pred).corr(pd.Series(true), method="spearman"))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": (
            float(1 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "pearson": pearson,
        "spearman": spearman,
        "mean_delta": float(error.mean()),
    }


def evaluate(model, loader, device, age_mean, age_std, amp_dtype=None):
    model.eval()
    predictions, targets, sexes, sites = [], [], [], []
    with torch.no_grad():
        for volume, age, metadata in tqdm(loader, desc="eval", leave=False):
            volume = volume.to(device, non_blocking=True)
            amp = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype is not None
                else contextlib.nullcontext()
            )
            with amp:
                pred = model(volume)
            predictions.extend(
                (pred.float().cpu().numpy() * age_std + age_mean).tolist()
            )
            targets.extend((age.numpy() * age_std + age_mean).tolist())
            sexes.extend(metadata["sex"])
            sites.extend(metadata["site"])
    metrics = regression_metrics(predictions, targets)
    metrics["predictions"] = predictions
    metrics["targets"] = targets
    metrics["sex"] = sexes
    metrics["site"] = sites
    return metrics


def print_metrics(label, metrics):
    fields = ["mae", "rmse", "r2", "pearson", "spearman", "mean_delta"]
    print(label + " | " + " | ".join(f"{key}={metrics[key]:.4f}" for key in fields))


def print_subgroup_metrics(metrics):
    frame = pd.DataFrame(
        {
            "prediction": metrics["predictions"],
            "target": metrics["targets"],
            "sex": metrics["sex"],
            "site": metrics["site"],
        }
    )
    frame["abs_error"] = (frame.prediction - frame.target).abs()
    print(
        "[test] MAE by sex:", frame.groupby("sex").abs_error.mean().round(3).to_dict()
    )
    frame["age_band"] = pd.cut(frame.target, bins=[0, 18, 30, 45, 60, float("inf")])
    print(
        "[test] MAE by age band:",
        frame.groupby("age_band", observed=True).abs_error.mean().round(3).to_dict(),
    )
    site_mae = frame.groupby("site").abs_error.agg(["mean", "count"])
    print("[test] site MAE range (sites with >=5 cases):", end=" ")
    eligible = site_mae[site_mae["count"] >= 5]["mean"]
    print(f"{eligible.min():.3f}-{eligible.max():.3f}" if len(eligible) else "n/a")


def make_scheduler(optimizer, warmup_steps, total_steps):
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.unfreeze_encoder:
        args.stage = "full"
    elif args.use_lora:
        args.stage = "lora"

    train_frame = OpenBHBAgeDataset.load_metadata(args.root, "train", args.modality)
    test_root = args.test_root or args.root
    test_frame = OpenBHBAgeDataset.load_metadata(test_root, "test", args.modality)
    overlap = set(train_frame.participant_id) & set(test_frame.participant_id)
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} IDs overlap train and test")
    train_ids, val_ids = make_train_val_split(train_frame, args.val_fraction, args.seed)
    age_mean, age_std = OpenBHBAgeDataset.compute_age_stats(
        args.root, "train", args.modality, train_ids
    )

    common = dict(
        modality=args.modality,
        target_size=tuple(args.target_size),
        age_mean=age_mean,
        age_std=age_std,
        return_metadata=True,
    )
    train_ds = OpenBHBAgeDataset(
        args.root, "train", participant_ids=train_ids, **common
    )
    val_ds = OpenBHBAgeDataset(args.root, "train", participant_ids=val_ids, **common)
    test_ds = OpenBHBAgeDataset(test_root, "test", **common)
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=generator, drop_last=False, **loader_args
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    vision_dir = resolve_vision_model_dir(args.vision_model_dir, args.model_cache)
    model = BrainAgeRegressor(
        vision_dir,
        depth=args.depth,
        depth_stride=args.depth_stride,
        max_depth_patches=args.max_depth_patches,
        stage=args.stage,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    ).to(device)
    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler = make_scheduler(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported() and not args.fp16
    amp_dtype = (
        torch.bfloat16 if bf16 else (torch.float16 if device.type == "cuda" else None)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    best_path = Path(args.output_dir) / "best.pt"
    last_path = Path(args.output_dir) / "last.pt"
    start_epoch, best_val, stale_epochs = 1, float("inf"), 0

    if args.resume:
        payload = load_checkpoint(
            args.resume, model, optimizer, scheduler, restore_rng=True
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = float(payload.get("best_val_mae", float("inf")))

    logger = ExperimentLogger(args)
    global_step = max(0, scheduler.last_epoch)

    print("=" * 72)
    print(f"device={device} amp={amp_dtype} stage={args.stage} loss={args.loss}")
    print(
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} age={age_mean:.2f}+/-{age_std:.2f}"
    )
    print(
        f"input={tuple(args.target_size)} tubelet={args.depth}/{args.depth_stride} max_depth_patches={args.max_depth_patches}"
    )
    print(
        f"trainable={sum(p.numel() for p in trainable)/1e6:.3f}M batch={args.batch_size} accum={args.grad_accum}"
    )
    print(
        f"mean predictor val MAE={np.mean(np.abs(val_ds.frame.age-age_mean)):.3f} years"
    )
    print("=" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        started = time.perf_counter()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch_index, (volume, age, _) in enumerate(bar):
            volume = volume.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            group_start = (batch_index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            amp = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else contextlib.nullcontext()
            )
            with amp:
                prediction = model(volume)
                raw_loss = criterion(prediction, age)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            end_group = (
                batch_index + 1
            ) % args.grad_accum == 0 or batch_index + 1 == len(train_loader)
            if end_group:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                logger.log(
                    {
                        "train/batch_loss": raw_loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    global_step,
                )
            running += raw_loss.item() * age.numel()
            seen += age.numel()
            bar.set_postfix(
                loss=f"{running/max(seen, 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        val_metrics = evaluate(model, val_loader, device, age_mean, age_std, amp_dtype)
        print_metrics(
            f"epoch={epoch} train_loss={running/max(seen, 1):.4f} time={time.perf_counter()-started:.1f}s val",
            val_metrics,
        )
        epoch_log = {"train/epoch_loss": running / max(seen, 1)}
        epoch_log.update(
            {
                f"validation/{key}": val_metrics[key]
                for key in ("mae", "rmse", "r2", "pearson", "spearman", "mean_delta")
            }
        )
        logger.log(epoch_log, global_step)
        improved = val_metrics["mae"] < best_val - args.min_delta
        if improved:
            best_val, stale_epochs = val_metrics["mae"], 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                args,
                (age_mean, age_std),
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val,
            args,
            (age_mean, age_std),
        )
        if stale_epochs >= args.patience:
            print(f"[early-stop] no validation improvement for {stale_epochs} epochs")
            break

    load_checkpoint(best_path, model)
    test_metrics = evaluate(model, test_loader, device, age_mean, age_std, amp_dtype)
    print_metrics("FINAL TEST", test_metrics)
    logger.log(
        {
            f"test/{key}": test_metrics[key]
            for key in ("mae", "rmse", "r2", "pearson", "spearman", "mean_delta")
        },
        global_step + 1,
    )
    print_subgroup_metrics(test_metrics)
    results = {
        key: value
        for key, value in test_metrics.items()
        if key not in {"predictions", "targets", "sex", "site"}
    }
    results.update({"best_val_mae": best_val, "age_mean": age_mean, "age_std": age_std})
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(
        Path(args.output_dir) / "test_metrics.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)
    logger.close()


def build_argparser():
    parser = argparse.ArgumentParser(description="BrainGemma3D brain-age regression")
    parser.add_argument(
        "--root", default="/media/fred/FRED5TB/Einstein/Open_BHB_processado"
    )
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--vision-model-dir", required=True)
    parser.add_argument("--model-cache", default="models")
    parser.add_argument(
        "--modality", default="quasiraw_3d", choices=["quasiraw_3d", "vbm_3d"]
    )
    parser.add_argument("--target-size", type=int, nargs=3, default=[32, 112, 112])
    parser.add_argument("--output-dir", default="checkpoints/brainage")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--stage", choices=["head", "stem", "lora", "full"], default="head"
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-depth-patches", type=int, default=8)
    parser.add_argument(
        "--use-lora", action="store_true", help="Compatibility alias for --stage lora"
    )
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Compatibility alias for --stage full",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16", action="store_true", help="Force FP16 instead of BF16"
    )
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="braingemma3d-brain-age")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
