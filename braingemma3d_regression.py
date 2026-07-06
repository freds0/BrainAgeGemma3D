#!/usr/bin/env python3
"""
BrainGemma3D - Brain Age Regression
===================================
Lean regressor for brain-age prediction on Open-BHB.

The MedGemma language model is intentionally NOT loaded: age prediction is a
pure image-regression task, so we keep only the 3D vision encoder
(``MedSigLIP3D``, reused from braingemma3d_architecture.py) and attach a small
MLP regression head.

Trainability (see BrainAgeRegressor):
  - freeze_encoder=True  -> SigLIP transformer weights frozen.
  - train_patch_embed=True -> the inflated 3D patch stem stays trainable
        (it is freshly inflated from 2D at init and otherwise untrained).
  - use_lora=True        -> LoRA adapters on the ViT attention (PEFT).
The regression head is always trainable.

Requires local MedSigLIP/SigLIP weights (a directory) OR a HF hub id passed via
--vision-model-dir; the gated weights are not bundled with this repo.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from braingemma3d_architecture import MedSigLIP3D
from braingemma3d_age_dataset import OpenBHBAgeDataset

VISION_DIM = 1152  # MedSigLIP hidden size


def resolve_vision_model_dir(spec: str, cache_root: str = "models") -> str:
    """
    Return a local directory holding the vision-model weights.

    ``spec`` may be an existing local dir (returned as-is) or a HuggingFace hub
    id (e.g. "google/medsiglip-448"), which is snapshot-downloaded into
    ``cache_root/<repo>``. MedSigLIP is gated: log in first (``hf auth login``
    or export HF_TOKEN) and accept the model licence on its HF page.
    ``load_sigclip_local`` uses local_files_only=True, so we must materialize a
    directory before constructing the model.
    """
    if os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download
    local_dir = os.path.join(cache_root, spec.replace("/", "__"))
    print(f"[weights] downloading '{spec}' -> {local_dir}")
    snapshot_download(
        repo_id=spec,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
    )
    return local_dir


# ============================================================
# MODEL
# ============================================================

class BrainAgeRegressor(nn.Module):
    """3D vision encoder + MLP head predicting a scalar age."""

    def __init__(
        self,
        vision_model_dir: str,
        depth: int = 2,
        freeze_encoder: bool = True,
        train_patch_embed: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        head_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MedSigLIP3D(model_name_or_dir=vision_model_dir, depth=depth)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            if train_patch_embed:
                # 3D patch stem is inflated-from-2D and untrained -> keep it learnable.
                for p in self.encoder.vision_model.patch_embedding_3d.parameters():
                    p.requires_grad = True

        if use_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            # Adapt the SigLIP transformer blocks only.
            self.encoder.vision_model.encoder = get_peft_model(
                self.encoder.vision_model.encoder, cfg
            )

        self.head = nn.Sequential(
            nn.LayerNorm(VISION_DIM),
            nn.Linear(VISION_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,D,H,W) -> (B,) predicted (normalized) age."""
        tokens = self.encoder.encode_image(x)      # (B,N,VISION_DIM)
        feat = tokens.mean(dim=1)                  # global average pool
        return self.head(feat.float()).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================
# CHECKPOINTING
# ============================================================

def save_regressor(model: BrainAgeRegressor, out_dir: str, extra: dict = None):
    """Persist only trainable tensors (head + patch stem + any LoRA)."""
    os.makedirs(out_dir, exist_ok=True)
    # Save only the trainable pieces: head + 3D patch stem + any LoRA adapters.
    state = {
        k: v.cpu() for k, v in model.state_dict().items()
        if k.startswith("head.") or "patch_embedding_3d" in k or "lora_" in k.lower()
    }
    payload = {"state": state}
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(out_dir, "regressor.pt"))
    print(f"[save] {len(state)} tensors -> {out_dir}/regressor.pt")


def load_regressor(model: BrainAgeRegressor, ckpt_path: str, map_location="cpu"):
    """Load a saved regressor checkpoint (strict=False: only trainable keys)."""
    payload = torch.load(ckpt_path, map_location=map_location)
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    print(f"[load] {ckpt_path} | unexpected={len(unexpected)}")
    return payload


# ============================================================
# TRAIN / EVAL
# ============================================================

def evaluate(model, loader, device, age_std, age_mean):
    """Return MAE in years on a loader."""
    model.eval()
    abs_err = 0.0
    n = 0
    with torch.no_grad():
        for vol, age in loader:
            vol = vol.to(device)
            pred = model(vol).cpu()
            pred_years = pred * age_std + age_mean
            true_years = age * age_std + age_mean
            abs_err += (pred_years - true_years).abs().sum().item()
            n += age.numel()
    return abs_err / max(n, 1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train and test may live under the same root (full Open-BHB) or under two
    # separate sample dirs (openbhb_train_sample / openbhb_test_sample), in which
    # case --test-root points at the second one.
    test_root = args.test_root or args.root
    age_mean, age_std = OpenBHBAgeDataset.compute_age_stats(args.root, "train")
    print(f"[data] age_mean={age_mean:.2f} age_std={age_std:.2f}")

    train_ds = OpenBHBAgeDataset(args.root, "train", args.modality,
                                 target_size=tuple(args.target_size),
                                 age_mean=age_mean, age_std=age_std)
    test_ds = OpenBHBAgeDataset(test_root, "test", args.modality,
                                target_size=tuple(args.target_size),
                                age_mean=age_mean, age_std=age_std)

    drop_last = len(train_ds) >= 2 * args.batch_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=drop_last)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    vision_dir = resolve_vision_model_dir(args.vision_model_dir, args.model_cache)
    model = BrainAgeRegressor(
        vision_model_dir=vision_dir,
        depth=args.depth,
        freeze_encoder=not args.unfreeze_encoder,
        train_patch_embed=not args.no_train_patch_embed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    if args.resume:
        load_regressor(model, args.resume, map_location=device)

    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[model] trainable params: {n_train/1e6:.3f}M")

    # MAE (L1) by default; swap to nn.MSELoss() for MSE.
    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    optim = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    best_mae = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze_encoder:
            model.encoder.eval()  # keep frozen BN/LN stats stable
        run = 0.0
        for i, (vol, age) in enumerate(train_loader):
            vol, age = vol.to(device), age.to(device)
            pred = model(vol)
            loss = criterion(pred, age) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                optim.step()
                optim.zero_grad()
            run += loss.item() * args.grad_accum
        train_loss = run / max(len(train_loader), 1)

        mae = evaluate(model, test_loader, device, age_std, age_mean)
        print(f"epoch {ep}/{args.epochs} | train_{args.loss}={train_loss:.4f} "
              f"| test_MAE={mae:.3f} yr")

        if mae < best_mae:
            best_mae = mae
            save_regressor(model, args.output_dir,
                           extra={"age_mean": age_mean, "age_std": age_std,
                                  "epoch": ep, "mae": mae})
    print(f"\nBest test MAE: {best_mae:.3f} years")


def build_argparser():
    ap = argparse.ArgumentParser(description="BrainGemma3D brain-age regression")
    ap.add_argument("--root", default="/media/fred/FRED5TB/Einstein/Open_BHB_processado",
                    help="Train root: has train.tsv and train/<modality>/")
    ap.add_argument("--test-root", default=None,
                    help="Test root if separate from --root (has test.tsv and "
                         "test/<modality>/). Defaults to --root.")
    ap.add_argument("--vision-model-dir", required=True,
                    help="Local MedSigLIP/SigLIP dir or HF hub id (e.g. google/medsiglip-448)")
    ap.add_argument("--model-cache", default="models",
                    help="Where hub weights are downloaded (project-local)")
    ap.add_argument("--modality", default="quasiraw_3d", choices=["quasiraw_3d", "vbm_3d"])
    ap.add_argument("--target-size", type=int, nargs=3, default=[64, 128, 128])
    ap.add_argument("--output-dir", default="checkpoints/brainage")
    ap.add_argument("--resume", default=None)

    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--unfreeze-encoder", action="store_true",
                    help="Train the whole ViT (default: frozen)")
    ap.add_argument("--no-train-patch-embed", action="store_true",
                    help="Also freeze the 3D patch stem (default: it stays trainable)")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)

    ap.add_argument("--loss", default="l1", choices=["l1", "mse"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
