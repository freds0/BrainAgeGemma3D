#!/usr/bin/env python3
"""
BrainGemma3D + LIME (3D) end-to-end CLI script

Pipeline:
1) Load trained BrainGemma3D from checkpoint (projector + optional LoRA)
2) Load one 3D NIfTI volume
3) Generate a reference report (report_ref) from the model (temperature=0)
4) Run LIME over 3D supervoxels (brain-only) using NLL score of report_ref tokens
5) Build a 3D weight volume (wvol)
6) SAVE overlay images to disk (NO plt.show)

Requirements (besides your project files):
- pip install lime scikit-image scipy matplotlib
"""

import os
import sys
import re
import argparse
from pathlib import Path
import json
import nibabel as nib

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

# Headless matplotlib (no display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Import project components
# ----------------------------
try:
    from braingemma3d_architecture import BrainGemma3D, load_nifti_volume, CANONICAL_PROMPT
    from braingemma3d_training import (
        set_seed,
        build_balanced_dataset,
        make_group_split,
        save_volume_slices,
    ) 
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Ensure braingemma3d_architecture.py and braingemma3d_training.py are accessible.")
    sys.exit(1)

# ----------------------------
# LIME + segmentation deps
# ----------------------------
from lime import lime_image
from skimage.segmentation import slic
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, binary_erosion
from skimage.morphology import ball, remove_small_objects
from skimage.measure import label as cc_label


# ============================================================
# 1) MODEL LOADING
# ============================================================
def load_trained_model(
    checkpoint_dir: str,
    vision_model_dir: str,
    language_model_dir: str,
    depth: int = 2,
    max_depth_patches: int = 128,
    num_vision_tokens: int = 32,
    device_map=None,
) -> BrainGemma3D:
    """Create base model and load trained weights (projector + vis_scale + optional LoRA).
    
    If checkpoint_dir is None or empty, loads BASE BrainGemma3D (no fine-tuning).
    """
    print(f"📥 Loading BrainGemma3D base model...", flush=True)
    model = BrainGemma3D(
        vision_model_dir=vision_model_dir,
        language_model_dir=language_model_dir,
        depth=depth,
        num_vision_tokens=num_vision_tokens,
        freeze_vision=True,
        freeze_language=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )

    # Handle checkpoint_dir = None -> use base model (no fine-tuning)
    if checkpoint_dir is None or checkpoint_dir == "" or not os.path.exists(checkpoint_dir):
        print("⚠️  No checkpoint directory provided or not found.", flush=True)
        print("    Using BASE BrainGemma3D (random projector, no LoRA).", flush=True)
        print("    ⚠️  LIME will run, but interpretability will NOT be clinically meaningful!", flush=True)
        model.eval()
        return model

    # Projector + vis_scale
    proj_path = os.path.join(checkpoint_dir, "projector_vis_scale.pt")
    if os.path.exists(proj_path):
        ckpt = torch.load(proj_path, map_location=model.lm_device)
        model.vision_projector.load_state_dict(ckpt["vision_projector"])
        if "vis_scale" in ckpt and ckpt["vis_scale"] is not None:
            val = ckpt["vis_scale"]
            if isinstance(val, torch.Tensor):
                model.vis_scale.data = val.to(model.lm_device)
            else:
                model.vis_scale.data.fill_(float(val))
        print(f"✅ Loaded projector | vis_scale={model.vis_scale.item():.3f}", flush=True)
    else:
        print(f"⚠️  Projector checkpoint not found at: {proj_path}", flush=True)
        print("    (LIME will still run, but vision-text alignment may be poor.)", flush=True)

    # Optional LoRA
    lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.isdir(lora_dir):
        # Verifica che esistano i file essenziali per PEFT
        adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"📎 Loading LoRA adapters from {lora_dir} ...", flush=True)
            try:
                from peft import PeftModel
                model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir, is_trainable=False)
                print("✅ Loaded LoRA adapters", flush=True)
            except Exception as e:
                print(f"⚠️  Failed to load LoRA adapters: {e}", flush=True)
                print("    Continuing with base language model.", flush=True)
        else:
            print(f"⚠️  LoRA directory exists but missing adapter_config.json: {lora_dir}", flush=True)
            print("    Skipping LoRA loading. Continuing with base language model.", flush=True)
    else:
        print("ℹ️  No LoRA adapters found (OK).", flush=True)

    model.eval()
    return model


# ============================================================
# 2) LIME SCORE = -NLL(report_ref | vision + prompt)
# ============================================================
@torch.no_grad()
def lime_score_report_nll(volumes, model, prompt: str, report_ref: str, batch_size: int = 1):
    """
    Score per volume = - average NLL on report_ref tokens,
    conditioned on [vision_tokens] + [prompt + report_ref].

    Output: (N, 1)
    """
    device = model.lm_device

    # 1) Tokenize prompt + report
    prompt_ids = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)       # (1,P)
    report_ids = model.tokenizer(report_ref, return_tensors="pt", add_special_tokens=False).input_ids.to(device) # (1,R)

    text_ids_1 = torch.cat([prompt_ids, report_ids], dim=1)  # (1, P+R)

    # 2) Prepare volumes tensor
    vols = torch.from_numpy(np.asarray(volumes)).to(device)
    if vols.ndim == 4:               # (N,Z,Y,X)
        vols = vols.unsqueeze(1)     # (N,1,Z,Y,X)

    N = vols.shape[0]
    scores = []

    for i in range(0, N, batch_size):
        v = vols[i:i+batch_size].to(dtype=torch.bfloat16)

        # 3) Encode volume -> vision tokens
        vision_tokens = model.encode_volume(v)  # (B, V, D_lm)

        # 4) Text embeddings
        text_ids = text_ids_1.repeat(v.size(0), 1)  # (B, P+R)
        text_embeds = model.language_model.get_input_embeddings()(text_ids)  # (B, P+R, D_lm)

        # 5) Concatenate embeds: [vision | text]
        inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)  # (B, V+P+R, D_lm)

        # 6) Labels: -100 su vision + prompt, target su report
        V = vision_tokens.size(1)
        prompt_mask = torch.full((v.size(0), prompt_ids.size(1)), -100, device=device, dtype=torch.long)
        vision_mask = torch.full((v.size(0), V), -100, device=device, dtype=torch.long)

        labels = torch.cat([vision_mask, prompt_mask, report_ids.repeat(v.size(0), 1)], dim=1)  # (B, V+P+R)

        # 7) Forward LM with labels -> loss
        out = model.language_model(inputs_embeds=inputs_embeds, labels=labels)
        loss = out.loss  # scalar mean over batch & tokens (masked)
        scores.append((-loss).detach().float().cpu())

    return torch.stack(scores).numpy().reshape(-1, 1)


# ============================================================
# 3) 3D SEGMENTATION: brain-only supervoxels
# ============================================================
def quick_brain_mask(
    vol_zyx: np.ndarray,
    p_thresh: float = 25,      # <-- percentile threshold (adatto a vol in [0,1])
    min_cc_vox: int = 2000
) -> np.ndarray:
    """
    vol_zyx: (Z,Y,X) float in [0,1] as returned by load_nifti_volume
    Returns boolean brain mask (Z,Y,X).
    """
    v = vol_zyx.astype(np.float32)

    # Robust threshold: separate background (near 0) from tissue
    thr = np.percentile(v, p_thresh)
    m = v > thr

    # morphology
    m = binary_opening(m, structure=ball(1))
    m = binary_closing(m, structure=ball(2))
    m = binary_fill_holes(m)

    # Remove small objects
    m = remove_small_objects(m, min_size=min_cc_vox)

    # Keep only the largest connected component
    cc = cc_label(m)
    if cc.max() > 1:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        m = cc == sizes.argmax()

    return m.astype(bool)


def big_supervoxels_brain_only(
    vol_zyx: np.ndarray,
    n_segments: int = 20,
    compactness: float = 0.05,
    sigma: float = 1.0,
    p_thresh: float = 25, 
    min_cc_vox: int = 2000,
):
    """
    Segment ONLY brain tissue using SLIC with brain mask.
    
    Returns segments with 0-based contiguous labels:
      - 0 = background (not brain)
      - 1, 2, ..., N = brain supervoxels
    
    This labeling is CRITICAL for LIME 0.2.0.1 which uses feature
    indices directly as segment labels: mask[segments == feature_idx].
    With 0-based contiguous labels, feature i maps exactly to segment i.
    Background (0) adds one harmless noise feature to LIME's regression.
    """
    brain = quick_brain_mask(vol_zyx, p_thresh=p_thresh, min_cc_vox=min_cc_vox)

    # Segment ONLY brain tissue using mask parameter.
    # Without mask, SLIC wastes most segments on empty background
    # (e.g. 84.5% background for typical BraTS volumes).
    seg = slic(
        vol_zyx,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        channel_axis=None,
        start_label=1,
        mask=brain,     # ← brain-only segmentation
    )
    # SLIC with mask assigns -1 to background voxels.
    # Relabel background to 0 for clean 0-based contiguous labels.
    seg[seg < 0] = 0

    # Verify labels are contiguous 0..N (required for LIME feature indexing).
    unique = np.unique(seg)
    expected = np.arange(len(unique))
    if not np.array_equal(unique, expected):
        new_seg = np.zeros_like(seg)
        for new_id, old_id in enumerate(unique):
            new_seg[seg == old_id] = new_id
        seg = new_seg
        print(f"ℹ️  Relabeled segments to contiguous 0..{len(unique)-1}", flush=True)

    n_brain_segs = len(np.unique(seg)) - 1  # exclude background (0)
    print(f"🧩 Brain-only SLIC: {n_brain_segs} brain supervoxels "
          f"(requested {n_segments}), brain covers {100*brain.sum()/brain.size:.1f}% of volume",
          flush=True)

    return seg, brain


def make_segmentation_fn(cached_segments: np.ndarray):
    """
    Return a segmentation_fn that always returns the pre-computed segments.
    
    This ensures LIME uses the EXACT SAME segmentation as the wvol
    construction. Previously, SLIC was called twice (inside LIME + after),
    which could produce different results.
    """
    def segmentation_fn(vol):
        return cached_segments
    return segmentation_fn

# ============================================================
# 4) VISUALIZATION HELPERS (SAVE TO FILE)
# ============================================================
def save_slice_png(volume_zyx: np.ndarray, out_path: str, axis: int = 0, idx: int | None = None, rot_k: int = 0):
    if idx is None:
        idx = volume_zyx.shape[axis] // 2

    if axis == 0:
        img = volume_zyx[idx, :, :]
        title = f"Axial (Z) slice {idx}"
    elif axis == 1:
        img = volume_zyx[:, idx, :]
        title = f"Coronal (Y) slice {idx}"
    else:
        img = volume_zyx[:, :, idx]
        title = f"Sagittal (X) slice {idx}"

    img = np.rot90(img, k=rot_k)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", origin="lower")
    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_overlay_png(
    volume_zyx: np.ndarray,
    heat_zyx: np.ndarray,
    out_path: str,
    axis: int = 0,
    idx: int | None = None,
    alpha: float = 0.45,
    clip_q: float = 0.99,
    rot_k: int = 0,
):
    assert volume_zyx.shape == heat_zyx.shape

    if idx is None:
        idx = volume_zyx.shape[axis] // 2

    if axis == 0:
        img = volume_zyx[idx, :, :]
        h = heat_zyx[idx, :, :]
        title = f"Axial (Z) overlay slice {idx}"
    elif axis == 1:
        img = volume_zyx[:, idx, :]
        h = heat_zyx[:, idx, :]
        title = f"Coronal (Y) overlay slice {idx}"
    else:
        img = volume_zyx[:, :, idx]
        h = heat_zyx[:, :, idx]
        title = f"Sagittal (X) overlay slice {idx}"

    img = np.rot90(img, k=rot_k)
    h = np.rot90(h, k=rot_k)

    m = float(max(np.quantile(np.abs(h), clip_q), 1e-8))
    h_vis = np.clip(h, -m, m)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", origin="lower")
    im = plt.imshow(h_vis, cmap="bwr", alpha=alpha, origin="lower", vmin=-m, vmax=m)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_overlay_grid_png(
    volume_zyx: np.ndarray,
    heat_zyx: np.ndarray,
    out_path: str,
    axis: int = 0,
    idxs=None,
    n_cols: int = 6,
    n_slices: int = 36,
    alpha: float = 0.45,
    clip_q: float = 0.99,
    rot_k: int = 0,
    figsize_per_cell: float = 2.2,
    add_colorbar: bool = False,
    suptitle: str | None = None,
):
    assert volume_zyx.shape == heat_zyx.shape
    assert axis in (0, 1, 2)

    dim = volume_zyx.shape[axis]
    if idxs is None:
        lo = int(0.10 * (dim - 1))
        hi = int(0.90 * (dim - 1))
        if hi <= lo:
            lo, hi = 0, dim - 1
        idxs = np.linspace(lo, hi, n_slices, dtype=int).tolist()
    else:
        idxs = list(map(int, idxs))

    n = len(idxs)
    n_rows = int(np.ceil(n / n_cols))

    m = float(max(np.quantile(np.abs(heat_zyx), clip_q), 1e-8))

    fig_w = n_cols * figsize_per_cell
    fig_h = n_rows * figsize_per_cell
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    def get_slice(arr, ax, i):
        if ax == 0:
            s = arr[i, :, :]
        elif ax == 1:
            s = arr[:, i, :]
        else:
            s = arr[:, :, i]
        return np.rot90(s, k=rot_k)

    im_for_cbar = None
    for j, idx in enumerate(idxs):
        axp = axes[j]
        img = get_slice(volume_zyx, axis, idx)
        h = get_slice(heat_zyx, axis, idx)
        h_vis = np.clip(h, -m, m)

        axp.imshow(img, cmap="gray", origin="lower")
        im_for_cbar = axp.imshow(h_vis, cmap="bwr", alpha=alpha, origin="lower", vmin=-m, vmax=m)
        axp.set_title(f"{idx}", fontsize=9)
        axp.axis("off")

    for k in range(n, n_rows * n_cols):
        axes[k].axis("off")

    if suptitle is None:
        name = "Axial (Z)" if axis == 0 else ("Coronal (Y)" if axis == 1 else "Sagittal (X)")
        suptitle = f"{name} | rot {rot_k*90}° | clip_q={clip_q} | alpha={alpha}"
    fig.suptitle(suptitle, y=0.98, fontsize=12)

    if add_colorbar and im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, ax=axes[:n], fraction=0.02, pad=0.01)
        cbar.set_label("LIME weight (clipped)", rotation=90)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def get_top_positive_supervoxel_id(weights: dict, ignore_ids=(0,)) -> int:
    """Return segment ID with highest positive LIME weight (most RED / supportive).
    Ignores background segment 0 by default."""
    items = [(int(k), float(v)) for k, v in weights.items() if int(k) not in ignore_ids]
    if not items:
        raise ValueError("weights empty or contains only ignored segments.")

    pos = [(k, v) for k, v in items if v > 0]
    if pos:
        return max(pos, key=lambda kv: kv[1])[0]
    return max(items, key=lambda kv: kv[1])[0]


def get_top_negative_supervoxel_id(weights: dict, ignore_ids=(0,)) -> int:
    """
    Returns the segment id with the most negative weight (most 'blue').
    If no negative weights exist, returns the minimum anyway (even if positive).
    """
    items = [(int(k), float(v)) for k, v in weights.items() if int(k) not in ignore_ids]
    if not items:
        raise ValueError("weights empty or contains only ignored segments.")

    neg = [(k, v) for k, v in items if v < 0]
    if neg:
        return min(neg, key=lambda kv: kv[1])[0]  # Most negative = minimum
    return min(items, key=lambda kv: kv[1])[0]


def _rgba_overlay_from_mask(mask2d: np.ndarray, rgba=(1.0, 0.0, 0.0), alpha=0.45) -> np.ndarray:
    """
    mask2d: float/bool (H,W) with 1 where to draw
    rgba: (R,G,B) in [0,1]
    """
    m = mask2d.astype(np.float32)
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = float(rgba[0])
    overlay[..., 1] = float(rgba[1])
    overlay[..., 2] = float(rgba[2])
    overlay[..., 3] = float(alpha) * m
    return overlay


def _rgba_edge_from_mask(mask2d: np.ndarray, rgba=(1.0, 0.0, 0.0), edge_alpha=1.0) -> np.ndarray:
    m = mask2d.astype(bool)
    edge = m & (~binary_erosion(m))
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = float(rgba[0])
    overlay[..., 1] = float(rgba[1])
    overlay[..., 2] = float(rgba[2])
    overlay[..., 3] = float(edge_alpha) * edge.astype(np.float32)
    return overlay


def save_overlay_single_supervoxel_png(
    volume_zyx: np.ndarray,
    segments_zyx: np.ndarray,
    weights: dict,
    out_path: str,
    axis: int = 0,
    idx: int | None = None,
    rot_k: int = 0,
    alpha: float = 0.45,
    origin: str = "lower",
    edge_alpha: float = 1.0,
):
    """
    Save overlay with:
      - Most 'red' supervoxel (maximum positive weight) in bright red
      - Most 'blue' supervoxel (most negative weight)   in bright blue
    Returns (best_red_id, best_blue_id).
    """
    best_red_id = get_top_positive_supervoxel_id(weights, ignore_ids=(0,))
    best_blue_id = get_top_negative_supervoxel_id(weights, ignore_ids=(0,))

    mask_red_3d = (segments_zyx == best_red_id).astype(np.float32)
    mask_blue_3d = (segments_zyx == best_blue_id).astype(np.float32)

    if idx is None:
        idx = volume_zyx.shape[axis] // 2

    # Extract slice (keep .T only for axial)
    if axis == 0:
        img = volume_zyx[idx, :, :]
        m_red = mask_red_3d[idx, :, :]
        m_blue = mask_blue_3d[idx, :, :]
        title = f"Axial(Z) slice {idx} | red={best_red_id} | blue={best_blue_id}"
    elif axis == 1:
        img = volume_zyx[:, idx, :]
        m_red = mask_red_3d[:, idx, :]
        m_blue = mask_blue_3d[:, idx, :]
        title = f"Coronal(Y) slice {idx} | red={best_red_id} | blue={best_blue_id}"
    else:
        img = volume_zyx[:, :, idx]
        m_red = mask_red_3d[:, :, idx]
        m_blue = mask_blue_3d[:, :, idx]
        title = f"Sagittal(X) slice {idx} | red={best_red_id} | blue={best_blue_id}"

    img = np.rot90(img, k=rot_k)
    m_red = np.rot90(m_red, k=rot_k)
    m_blue = np.rot90(m_blue, k=rot_k)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", origin=origin)

    # Blue first, red on top (so red wins if overlapping)
    plt.imshow(_rgba_overlay_from_mask(m_blue, rgba=(0.0, 0.4, 1.0), alpha=alpha), origin=origin)
    plt.imshow(_rgba_edge_from_mask(m_blue, rgba=(0.0, 0.4, 1.0), edge_alpha=edge_alpha), origin=origin)

    plt.imshow(_rgba_overlay_from_mask(m_red, rgba=(1.0, 0.0, 0.0), alpha=alpha), origin=origin)
    plt.imshow(_rgba_edge_from_mask(m_red, rgba=(1.0, 0.0, 0.0), edge_alpha=edge_alpha), origin=origin)

    plt.title(title)
    plt.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return best_red_id, best_blue_id


def save_overlay_grid_single_supervoxel_png(
    volume_zyx: np.ndarray,
    segments_zyx: np.ndarray,
    weights: dict,
    out_path: str,
    axis: int = 0,
    n_cols: int = 8,
    rot_k: int = 0,
    alpha: float = 0.45,
    origin: str = "lower",
    suptitle: str | None = None,
    edge_alpha: float = 1.0,
):
    """
    Grid overlay with ALL slices, organized like save_flair_grid_all:
      - most 'red' supervoxel in bright red
      - most 'blue' supervoxel in bright blue
    Returns (best_red_id, best_blue_id).
    """
    best_red_id = get_top_positive_supervoxel_id(weights, ignore_ids=(0,))
    best_blue_id = get_top_negative_supervoxel_id(weights, ignore_ids=(0,))

    mask_red_3d = (segments_zyx == best_red_id).astype(np.float32)
    mask_blue_3d = (segments_zyx == best_blue_id).astype(np.float32)

    dim = volume_zyx.shape[axis]
    n_rows = int(np.ceil(dim / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2, n_rows * 2),
        facecolor="black"
    )
    axes = np.array(axes).reshape(-1)

    def get_slice(arr, ax, i):
        if ax == 0:
            s = arr[i, :, :]
        elif ax == 1:
            s = arr[:, i, :]
        else:
            s = arr[:, :, i]
        return np.rot90(s, k=rot_k)

    for i in range(dim):
        img = get_slice(volume_zyx, axis, i)
        m_red = get_slice(mask_red_3d, axis, i)
        m_blue = get_slice(mask_blue_3d, axis, i)

        axes[i].imshow(img, cmap="gray", origin=origin)

        # blue below, red above
        axes[i].imshow(_rgba_overlay_from_mask(m_blue, rgba=(0.0, 0.4, 1.0), alpha=alpha), origin=origin)
        axes[i].imshow(_rgba_edge_from_mask(m_blue, rgba=(0.0, 0.4, 1.0), edge_alpha=edge_alpha), origin=origin)

        axes[i].imshow(_rgba_overlay_from_mask(m_red, rgba=(1.0, 0.0, 0.0), alpha=alpha), origin=origin)
        axes[i].imshow(_rgba_edge_from_mask(m_red, rgba=(1.0, 0.0, 0.0), edge_alpha=edge_alpha), origin=origin)

        axes[i].set_title(
            f"z={i}",
            color="cyan",
            fontsize=9,
            fontweight='bold'
        )
        axes[i].axis("off")

    # Turn off axes for any unused subplots
    for i in range(dim, len(axes)):
        axes[i].axis("off")

    if suptitle is None:
        name = "Axial(Z)" if axis == 0 else ("Coronal(Y)" if axis == 1 else "Sagittal(X)")
        suptitle = f"{name} | red={best_red_id} | blue={best_blue_id} | rot {rot_k*90}°"
    fig.suptitle(suptitle)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return best_red_id, best_blue_id


def save_volume_slices_overlay(
    vol: torch.Tensor,
    heat: np.ndarray,              # Heatmap as numpy array
    save_path: str,
    title: str = "Volume overlay",
    ncols: int = 8,
    is_healthy: bool = False,
    alpha: float = 0.45,
    clip_q: float = 0.99,
    rot_k: int = 0,
    brain_mask: np.ndarray | None = None,   # <--- aggiungi
):
    # --- squeeze to (D,H,W)
    if vol.ndim == 5:
        vol = vol[0, 0]
    elif vol.ndim == 4:
        vol = vol[0]

    vol_np = vol.detach().cpu().numpy().astype(np.float32)
    heat_np = heat.astype(np.float32)

    if vol_np.shape != heat_np.shape:
        raise ValueError(f"Shape mismatch: vol {vol_np.shape} vs heat {heat_np.shape}")

    if brain_mask is not None:
        if brain_mask.shape != vol_np.shape:
            raise ValueError(f"Brain mask shape mismatch: {brain_mask.shape} vs {vol_np.shape}")
        brain_np = brain_mask.astype(bool)
    else:
        brain_np = None

    D, H, W = vol_np.shape
    nrows = int(np.ceil(D / ncols))

    # clipping globale coerente
    m = float(max(np.quantile(np.abs(heat_np), clip_q), 1e-8))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), facecolor="black")
    axes = axes.flatten()

    for i in range(D):
        img = vol_np[i]
        h = heat_np[i]

        if brain_np is not None:
            b = brain_np[i]
        else:
            b = None

        # rotazione
        img = np.rot90(img, k=rot_k)
        h = np.rot90(h, k=rot_k)
        if b is not None:
            b = np.rot90(b, k=rot_k)

        # clip heat
        h_vis = np.clip(h, -m, m)

        ax = axes[i]
        ax.set_facecolor("black")

        if b is not None:
            # Mask img: outside brain -> transparent
            img_ma = np.ma.array(img, mask=~b)
            ax.imshow(img_ma, cmap="gray", origin="lower")

            # Mask heat too: outside brain -> transparent
            h_ma = np.ma.array(h_vis, mask=~b)
            ax.imshow(h_ma, cmap="bwr", alpha=alpha, vmin=-m, vmax=m, origin="lower")
        else:
            ax.imshow(img, cmap="gray", origin="lower")
            ax.imshow(h_vis, cmap="bwr", alpha=alpha, vmin=-m, vmax=m, origin="lower")

        ax.set_title(f"z={i}", color="cyan", fontsize=9, fontweight="bold")
        ax.axis("off")

    for i in range(D, len(axes)):
        axes[i].set_facecolor("black")
        axes[i].axis("off")

    fig.suptitle(f"{title} {'(Healthy)' if is_healthy else '(Pathological)'}", color="white")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_flair_grid_all(nifti_path: str, save_path: str, ncols: int = 8):
    vol = load_nifti_volume(nifti_path)
    vol = vol.squeeze(0).squeeze(0).detach().cpu().numpy()
    D = vol.shape[0]
    nrows = int(np.ceil(D / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2, nrows * 2),
        facecolor="black"
    )
    axes = axes.flatten()

    for i in range(D):
        axes[i].imshow(vol[i], cmap="gray", origin="lower")
        axes[i].set_title(
            f"z={i}",
            color="cyan",
            fontsize=9,
            fontweight='bold'
        )
        axes[i].axis("off")

    # Spegni assi inutilizzati
    for i in range(D, len(axes)):
        axes[i].axis("off")

    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5) MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser("BrainGemma3D + LIME interpretability (3D)")

    # Paths
    ap.add_argument("--checkpoint-dir", default=None, type=str, help="Checkpoint folder with fine-tuned weights (projector_vis_scale.pt + optional lora_adapters/). If None, uses BASE BrainGemma3D (no fine-tuning).")
    ap.add_argument("--base-dir", required=True, type=str, help="Base dir containing Models/siglip and Models/medgemma")
    ap.add_argument("--input-volume", default=None, type=str, help="Path to .nii or .nii.gz volume. If None, uses the test set.")
    ap.add_argument("--output-dir", default="lime_output", type=str, help="Where to save report + images")
    ap.add_argument("--evaluation-samples", default=None, type=str, help="CSV file with generated reports. If provided, uses these samples instead of reconstructing the test set.")

    # Model config (must match training)
    ap.add_argument("--num-vision-tokens", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-depth-patches", type=int, default=128)

    # Volume preprocessing
    ap.add_argument("--target-size", type=int, nargs=3, default=[64, 128, 128])

    # Prompt / generation for report_ref
    ap.add_argument("--prompt", type=str, default=None, help="If None uses CANONICAL_PROMPT. Supports \\n from CLI.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)  # for report_ref
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)

    # Dataset params
    ap.add_argument("--num-brats-patients", type=int, default=369) # Default used in training
    ap.add_argument("--num-healthy-patients", type=int, default=99) # Default used in training
    ap.add_argument("--modality", type=str, default="flair")

    # LIME params
    ap.add_argument("--lime-samples", type=int, default=100, help="num_samples for LIME")
    ap.add_argument("--lime-batch-size", type=int, default=1, help="Keep 1 for per-sample NLL score")
    ap.add_argument("--hide-color", type=float, default=0.0)

    # Segmentation params (supervoxels)
    ap.add_argument("--n-segments", type=int, default=20)
    ap.add_argument("--compactness", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--z-thresh", type=float, default=-0.3)
    ap.add_argument("--min-cc-vox", type=int, default=2000)

    # Visualization params
    ap.add_argument("--slice-idx", type=int, default=42, help="Slice index for single-slice overlays (axis=0 uses Z)")
    ap.add_argument("--rot-k", type=int, default=0, help="Rotate slices by k*90 degrees")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clip-q", type=float, default=0.99)
    ap.add_argument("--grid-cols", type=int, default=6)
    ap.add_argument("--grid-slices", type=int, default=36)

    # Seed
    ap.add_argument("--seed", type=int, default=42, help="Seed (match evaluation script)")

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # eval_csv
    eval_df = None

    if args.evaluation_samples is not None:
        csv_path = Path(args.base_dir) / args.evaluation_samples
        print(f"📄 Loading evaluation CSV from: {csv_path}", flush=True)
        
        if csv_path.exists():
            eval_df = pd.read_csv(csv_path)
            print(f"✅ CSV loaded: {len(eval_df)} rows", flush=True)
        else:
            print("⚠️ CSV not found. Will generate reports normally.", flush=True)

    # Resolve model dirs
    base_dir = Path(args.base_dir)
    vision_model_dir = base_dir / "Models" / "siglip"
    language_model_dir = base_dir / "Models" / "medgemma"

    if not vision_model_dir.exists():
        print(f"❌ Missing: {vision_model_dir}", flush=True)
        sys.exit(1)
    if not language_model_dir.exists():
        print(f"❌ Missing: {language_model_dir}", flush=True)
        sys.exit(1)

    # Device map
    device_map = {"": 0} if torch.cuda.is_available() else None
    print(f"🖥️  cuda_available={torch.cuda.is_available()} | device_map={device_map}", flush=True)

    # Load model
    model = load_trained_model(
        checkpoint_dir=args.checkpoint_dir,
        vision_model_dir=str(vision_model_dir),
        language_model_dir=str(language_model_dir),
        depth=args.depth,
        max_depth_patches=args.max_depth_patches,
        num_vision_tokens=args.num_vision_tokens,
        device_map=device_map,
    )

    # Prompt
    prompt = CANONICAL_PROMPT if args.prompt is None else args.prompt.replace("\\n", "\n")

    # 1. DATASET RECONSTRUCTION & SPLIT (La parte fondamentale)
    if args.input_volume is not None:
        if not os.path.exists(args.input_volume):
            print(f"❌ Missing input volume: {args.input_volume}", flush=True)
            sys.exit(1)
        print(f"⚠️  Input volume provided: {args.input_volume}")
        print("    Skipping dataset reconstruction and using the provided volume directly.")
        print("    Note: LIME scores may not be meaningful if the model was not trained on this data distribution.")
        # In this case, construct a dummy dataset with a single example (the provided volume)
        is_healthy = True if "healthy" in args.input_volume.lower() else False
        dataset = [{
            "image_path": args.input_volume,
            "patient_id": args.input_volume.split("/")[-1].split("_")[0] if is_healthy else args.input_volume.split("/")[-2],
            "report_path": None,
            "is_healthy": is_healthy,
        }]
        test_data = dataset
    else:
        print("\n📚 Reconstructing Dataset Logic (Seed 42)...")
        brats_images = str(Path(args.base_dir) / "Datasets" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData")
        brats_reports = str(Path(args.base_dir) / "Datasets" / "TextBraTS" / "TextBraTSData")
        healthy_brains = str(Path(args.base_dir) / "Datasets" / "HealthyBrains_Preprocessed")
        
        # Ricostruiamo l'intero dataset con la logica "balanced"
        dataset = build_balanced_dataset(
            brats_images_base=brats_images,
            brats_reports_base=brats_reports,
            healthy_brains_base=healthy_brains,
            num_brats_patients=args.num_brats_patients,
            num_healthy_patients=args.num_healthy_patients,
            modality=args.modality,
        )
        
        # Perform 70/10/20 split exactly as in training
        _, _, test_data = make_group_split(dataset, seed=args.seed, train_frac=0.7, val_frac=0.1)
        
        print(f"✅ Isolated Test Set: {len(test_data)} patients")
        print(f"   Pathological (BraTS): {sum(1 for x in test_data if not x['is_healthy'])}")
        print(f"   Healthy:              {sum(1 for x in test_data if x['is_healthy'])}")

    def safe_name(s: str) -> str:
        s = str(s)
        # Replace all non-alphanumeric characters (except _, -, .) with _
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

    for i, ex in enumerate(test_data):
        patient_id = safe_name(ex.get("patient_id", f"sample_{i:04d}"))

        # ✅ sottocartella per soggetto
        subj_dir = out_dir / patient_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Load volume
        print("🧠 Loading NIfTI volume at path ", ex["image_path"], flush=True)
        volume = load_nifti_volume(ex["image_path"], target_size=tuple(args.target_size))
        volume = volume.to(model.lm_device)
        if volume.ndim == 4:
            volume = volume.unsqueeze(0)  # Batch dim
        vol_np = volume[0, 0].detach().cpu().numpy().astype(np.float32)  # (Z,Y,X)
        print(f"✅ Volume loaded: {vol_np.shape}", flush=True)
        print(f"📁 Subject output dir: {subj_dir}", flush=True)

        # Save volume slices
        save_path = os.path.join(subj_dir, f"input_grid.png")
        is_healthy_data = ex.get('is_healthy', False) or "HealthyBrains" in ex.get("image_path", "") or "healthy" in ex.get("image_path", "").lower()
        save_volume_slices(volume, save_path, title=f"{patient_id} - INPUT", is_healthy=is_healthy_data)

        # Generate reference report
        report_ref = None

        if eval_df is not None:            
            matches = eval_df[eval_df["patient_id"] == patient_id]
            if len(matches) > 0:
                print("📄 Found existing generated report in CSV. Using that.", flush=True)
                report_ref = matches.iloc[0]["gen"]

        if report_ref is None:
            print("✍️  Generating reference report (report_ref) ...", flush=True)
            with torch.no_grad():
                report_ref = model.generate_report(
                    volume,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )

        (subj_dir / "report_ref.txt").write_text(report_ref, encoding="utf-8")
        print(f"✅ Saved report_ref to: {subj_dir / 'report_ref.txt'}", flush=True)
        print(f"***** GENERATED REPORT *******\n{report_ref}", flush=True)
        print("******************************\n", flush=True)

        # LIME explainer
        # ── 1) Compute segments ONCE (shared between LIME and wvol) ──
        print("🧩 Computing brain-only supervoxels...", flush=True)
        segments, brain = big_supervoxels_brain_only(
            vol_np,
            n_segments=args.n_segments,
            compactness=args.compactness,
            sigma=args.sigma,
            p_thresh=25,
            min_cc_vox=args.min_cc_vox,
        )
        # Segmentation_fn returns cached segments (no second SLIC call)
        segmentation_fn = make_segmentation_fn(segments)

        print("🧪 Running LIME (this can be slow)...", flush=True)
        explainer = lime_image.LimeImageExplainer(random_state=args.seed)
        
        # Increase batch size for speed (A100 can handle 16/32 easily)
        def predict_fn(vols):
            return lime_score_report_nll(
                vols,
                model=model,
                prompt=prompt,
                report_ref=report_ref,
                batch_size=1,  
            )

        explanation = explainer.explain_instance(
            vol_np,
            classifier_fn=predict_fn,
            segmentation_fn=segmentation_fn,
            labels=[0],
            hide_color=float(args.hide_color),
            num_samples=int(args.lime_samples),
        )

        # ── 3) Build wvol from weights ──
        # segments already computed above (same as LIME used)
        weights = dict(explanation.local_exp[np.int64(0)])

        wvol = np.zeros_like(vol_np, dtype=np.float32)
        for seg_id, w in weights.items():
            seg_id = int(seg_id)
            if seg_id == 0:    # Skip background (segment 0)
                continue
            wvol[segments == seg_id] = float(w)

        # Safety: zero out anything outside brain mask
        wvol[~brain] = 0.0

        np.save(str(subj_dir / "lime_wvol.npy"), wvol)
        np.save(str(subj_dir / "lime_segments.npy"), segments)
        print(f"✅ Saved wvol/segments to: {subj_dir}", flush=True)
        print(f"   wvol stats: shape={wvol.shape} min={wvol.min():.4g} max={wvol.max():.4g}", flush=True)

        # Save overlays
        print("🖼️  Saving overlay images...", flush=True)

        save_volume_slices_overlay(
            volume,               # torch tensor
            wvol,                 # torch tensor heatmap
            str(subj_dir / "overlay_slices.png"),
            title="Interpretability",
            ncols=8,
            is_healthy=False,
            alpha=args.alpha,
            clip_q=args.clip_q,
            rot_k=args.rot_k,
            brain_mask=brain,
        )

        save_overlay_grid_single_supervoxel_png(
            vol_np, segments, weights,
            out_path=str(subj_dir / "best_supervoxel_grid_axial.png"),
            axis=0, n_cols=8, rot_k=0, alpha=0.55
        )
        
        # Save supervoxel weights in JSON format (easy to read/analyze)
        # Exclude background segment (0) from saved weights
        weights_dict = {int(k): float(v) for k, v in weights.items() if int(k) != 0}
        with open(subj_dir / "lime_weights.json", "w") as f:
            json.dump(weights_dict, f, indent=2)
        print(f"💾 Saved lime_weights.json ({len(weights_dict)} brain supervoxels)", flush=True)

    print("✅ Done.", flush=True)
    print(f"📁 Outputs in: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()