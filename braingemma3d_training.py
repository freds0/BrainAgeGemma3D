#!/usr/bin/env python3
"""
BrainGemma3D Training Script
=============================
Complete training pipeline with:
- Balanced dataset (pathological BraTS + healthy brains)
- Healthy reports: randomized variants to prevent shortcut learning
- CANONICAL PROMPT immutable (NO augmentation, NO system instruction)
- Phase 1: Image-Text Grounding (NO prompt, MRI → report only)
- Phase 2A/2B: Report generation with the same canonical prompt
- Shuffle per epoch (NO weighted sampling, natural distribution)
- Group split to avoid data leakage
"""

# ENVIRONMENT CONFIGURATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import sys
import argparse
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# IMPORT MODEL and UTILS
from braingemma3d_architecture import (
    BrainGemma3D,
    load_nifti_volume,
    get_volume_from_ex,
    CANONICAL_PROMPT,
    get_training_prompt,
    get_inference_prompt,
)

from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# SEED / DEVICE
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# HEALTHY BRAIN TEMPLATES
# ============================================================

HEALTHY_TEMPLATES = {
    "flair": [
        "The brain parenchyma appears within normal limits on this FLAIR MRI.",
        "The brain shows normal signal characteristics on FLAIR sequence.",
        "No focal abnormalities are identified on this FLAIR MRI examination.",
        "No definite focal abnormality is identified in the provided FLAIR scan.",
        "The FLAIR MRI demonstrates unremarkable brain parenchyma.",
        "Normal brain appearance is noted on this FLAIR sequence.",
        "The brain parenchyma shows expected signal on FLAIR imaging.",
    ],
}


def get_healthy_response(modality: str = "flair", patient_id: str = None, seed: int = 0) -> str:
    """
    Response for healthy brains - 1-2 vague sentences, neutral clinical language.
    Seed controlled via patient_id for determinism.
    """
    m = modality.lower()
    templates = HEALTHY_TEMPLATES.get(m, HEALTHY_TEMPLATES["flair"])
    
    if patient_id is not None:
        patient_hash = abs(hash(patient_id + str(seed)))
        idx = patient_hash % len(templates)
    else:
        idx = seed % len(templates)
    
    return templates[idx]


def clean_report_text(text: str) -> str:
    """Clean the report text while keeping full content."""
    text = ' '.join(text.split())
    if text and not text.endswith('.'):
        text += '.'
    return text.strip()


def extract_lesion_side(report_text: str) -> str:
    """
    Extracts lesion side from the textual report.
    
    Returns:
        - "left" if lesion is on the left
        - "right" if lesion is on the right
        - "bilateral" if bilateral
        - "unknown" if not specified
    """
    text_lower = report_text.lower()
    
    # Pattern for bilateral
    bilateral_patterns = [
        "bilateral", "bilaterally", "both hemisphere", "both sides",
        "left and right", "right and left"
    ]
    
    # Pattern for left
    left_patterns = [
        "left hemisphere", "left frontal", "left temporal", "left parietal",
        "left occipital", "left cerebellar", "left basal ganglia", "left thalamus",
        "left ventricle", "left side", "in the left", "on the left"
    ]
    
    # Pattern for right
    right_patterns = [
        "right hemisphere", "right frontal", "right temporal", "right parietal",
        "right occipital", "right cerebellar", "right basal ganglia", "right thalamus",
        "right ventricle", "right side", "in the right", "on the right"
    ]
    
    # Check bilateral first
    if any(pattern in text_lower for pattern in bilateral_patterns):
        return "bilateral"
    
    # Check left and right
    has_left = any(pattern in text_lower for pattern in left_patterns)
    has_right = any(pattern in text_lower for pattern in right_patterns)
    
    if has_left and has_right:
        return "bilateral"
    elif has_left:
        return "left"
    elif has_right:
        return "right"
    else:
        return "unknown"


# ============================================================
# DATASET LOADING
# ============================================================

def build_balanced_dataset(
    brats_images_base: str,
    brats_reports_base: str,
    healthy_brains_base: str,
    num_brats_patients: Optional[int] = None,
    num_healthy_patients: Optional[int] = None,
    modality: str = "flair",
) -> List[Dict]:
    """
    Create balanced dataset with:
    - BraTS patients (pathological)
    - Healthy controls from OpenNeuro
    """
    
    dataset = []
    
    # 1) BRATS PATHOLOGICAL BRAINS
    print(f"📚 Loading BraTS pathological dataset...")
    brats_base = Path(brats_images_base)
    reports_base = Path(brats_reports_base)
    
    if not brats_base.exists():
        raise FileNotFoundError(f"BraTS directory not found: {brats_base}")
    
    patient_folders = sorted([d for d in brats_base.iterdir() if d.is_dir()])
    if num_brats_patients:
        patient_folders = patient_folders[:num_brats_patients]
    
    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_{modality}.nii"
        
        if not img_path.exists():
            continue
        
        report_path = reports_base / patient_id / f"{patient_id}_{modality}_text.txt"
        if not report_path.exists():
            continue
        
        report_text = ""
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                report_text = f.read().strip()
                report_text = clean_report_text(report_text)
        
        if len(report_text) < 20:
            continue
        
        # Extract lesion side for stratification
        lesion_side = extract_lesion_side(report_text)
        
        dataset.append({
            "image_path": str(img_path),
            "report": report_text,
            "patient_id": patient_id,
            "task_type": "generate_report",
            "is_healthy": False,
            "modality": modality,
            "lesion_side": lesion_side,
        })
    
    print(f"✅ Loaded {len(dataset)} BraTS patients")
    
    # 2) HEALTHY BRAINS PREPROCESSED
    print(f"🧠 Loading healthy brain controls (PREPROCESSED)...")
    healthy_base = Path(healthy_brains_base)
    
    # Skip healthy brains if num_healthy_patients is explicitly set to 0
    if num_healthy_patients == 0:
        print(f"⏭️  Skipping healthy brains (num_healthy_patients=0)")
    elif healthy_base.exists():
        preprocessed_files = sorted(healthy_base.glob("sub-*_preprocessed.nii*"))
        
        if num_healthy_patients is not None:
            preprocessed_files = preprocessed_files[:num_healthy_patients]
        
        healthy_count = 0
        for img_path in preprocessed_files:
            subject_id = img_path.stem.split('_preprocessed')[0]
            if img_path.suffix == '.gz':
                subject_id = subject_id.rsplit('.', 1)[0]
            
            dataset.append({
                "image_path": str(img_path),
                "report": get_healthy_response(modality, patient_id=subject_id, seed=0),
                "patient_id": subject_id,
                "task_type": "generate_report",
                "is_healthy": True,
                "modality": modality,
                "lesion_side": "none",  # Healthy brains have no lesions
            })
            healthy_count += 1
        
        print(f"✅ Loaded {healthy_count} healthy brain controls (PREPROCESSED)")
    else:
        print(f"⚠️  Healthy brains directory not found: {healthy_base}")
        print(f"   Training will proceed with BraTS only (no healthy controls)")
    
    n_healthy = sum(1 for ex in dataset if ex["is_healthy"])
    n_pathological = len(dataset) - n_healthy
    
    # Statistics for lesion side (pathological only)
    lesion_sides = [ex["lesion_side"] for ex in dataset if not ex["is_healthy"]]
    side_counts = {}
    for side in lesion_sides:
        side_counts[side] = side_counts.get(side, 0) + 1
    
    print(f"\n📊 Dataset composition:")
    print(f"   Pathological (BraTS): {n_pathological}")
    print(f"   Healthy (OpenNeuro): {n_healthy}")
    print(f"   Total: {len(dataset)}")
    print(f"   Healthy ratio: {n_healthy/len(dataset)*100:.1f}%")
    print(f"   Pathological:healthy ratio = {n_pathological/max(n_healthy,1):.1f}:1")
    print(f"\n📍 Lesion side distribution (pathological only):")
    for side, count in sorted(side_counts.items()):
        print(f"   {side.capitalize()}: {count} ({count/max(n_pathological,1)*100:.1f}%)")
    
    return dataset


def make_group_split(
    dataset: List[Dict],
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    group_key: str = "image_path",
    stratify_by_lesion_side: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Group split to avoid data leakage with stratification by lesion_side.
    Splits by groups while keeping balance between left/right/bilateral/healthy.
    """
    random.seed(seed)
    
    if stratify_by_lesion_side:
        # Stratify by lesion_side
        strata = defaultdict(list)
        for ex in dataset:
            side = ex.get("lesion_side", "unknown")
            strata[side].append(ex)
        
        print(f"\n📊 Stratified split by lesion_side:")
        for side, exs in strata.items():
            print(f"   {side.capitalize()}: {len(exs)} samples")
        
        # Split each stratum separately
        train_data, val_data, test_data = [], [], []
        
        for side, exs in strata.items():
            groups = defaultdict(list)
            for ex in exs:
                groups[ex[group_key]].append(ex)
            
            group_ids = list(groups.keys())
            random.shuffle(group_ids)
            
            n = len(group_ids)
            n_train = max(1, int(n * train_frac)) if n >= 3 else (1 if n >= 2 else n)
            n_val = max(1, int(n * val_frac)) if n >= 3 else (0 if n < 2 else 0)
            
            train_g = set(group_ids[:n_train])
            val_g = set(group_ids[n_train:n_train+n_val])
            test_g = set(group_ids[n_train+n_train+n_val:] if False else group_ids[n_train+n_val:])  # keep identical behavior
            
            for gid in train_g:
                train_data.extend(groups[gid])
            for gid in val_g:
                val_data.extend(groups[gid])
            for gid in test_g:
                test_data.extend(groups[gid])
        
        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
    else:
        # Normal split without stratification
        groups = defaultdict(list)
        for ex in dataset:
            groups[ex[group_key]].append(ex)
        
        group_ids = list(groups.keys())
        random.shuffle(group_ids)
    
        n = len(group_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        elif n == 2:
            n_train = 1
            n_val = 0
        
        train_g = set(group_ids[:n_train])
        val_g = set(group_ids[n_train:n_train+n_val])
        test_g = set(group_ids[n_train+n_train+n_val:] if False else group_ids[n_train+n_val:])
        
        def collect(gset):
            out = []
            for gid in gset:
                out.extend(groups[gid])
            return out
        
        train_data = collect(train_g)
        val_data = collect(val_g)
        test_data = collect(test_g)
        
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
    
    # Final statistics per split
    def side_stats(data, name):
        side_counts = {}
        for ex in data:
            side = ex.get("lesion_side", "unknown")
            side_counts[side] = side_counts.get(side, 0) + 1
        print(f"   {name}: {len(data)} samples")
        for side, count in sorted(side_counts.items()):
            print(f"      {side}: {count}")
    
    print(f"\n📊 Final split statistics:")
    side_stats(train_data, "Train")
    side_stats(val_data, "Val")
    side_stats(test_data, "Test")
    
    return train_data, val_data, test_data


# ============================================================
# PHASE 1 - ALIGNMENT
# ============================================================

@torch.no_grad()
def text_embed_from_lm(model: BrainGemma3D, texts: List[str], max_length: int = 256):
    """Extract text embeddings from the LM (mean pooling)"""
    tok = model.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    ).to(model.lm_device)

    out = model.language_model(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    h = out.hidden_states[-1]
    mask = tok.attention_mask.unsqueeze(-1)
    h = h * mask
    denom = mask.sum(dim=1).clamp(min=1)
    h = h.sum(dim=1) / denom
    return F.normalize(h.float(), p=2, dim=-1)


def encode_volume_for_alignment_phase1(model: BrainGemma3D, volume_3d: torch.Tensor):
    """
    Forward for Phase 1: patch_embedding_3d trainable, rest frozen.
    NO no_grad(), NO early pooling.
    """
    volume_3d = volume_3d.to(model.lm_device, dtype=torch.bfloat16)
    vision_model = model.vision_encoder.vision_model

    x = vision_model.patch_embedding_3d(volume_3d)
    B, E, Dp, Hp, Wp = x.shape
    x = x.flatten(2).transpose(1, 2)
    
    pos = vision_model.get_position_embedding_3d(Dp, Hp * Wp, Hp, Wp).to(x.device, dtype=x.dtype)
    x = x + pos
    x = vision_model.encoder(x).last_hidden_state
    x = vision_model.post_layernorm(x)

    x = x.mean(dim=1, keepdim=True)
    
    x = model.vision_projector(x.to(torch.bfloat16))
    x = x * model.vis_scale.to(x.dtype)
    return x


def encode_volume_for_alignment_no_grad_vit(model: BrainGemma3D, volume_3d: torch.Tensor):
    """
    Forward for Phase 2A/2B: everything in no_grad (ViT frozen) + projector with grad.
    Uses adaptive pooling to reduce tokens.
    """
    volume_3d = volume_3d.to(model.lm_device, dtype=torch.bfloat16)
    vision_model = model.vision_encoder.vision_model

    with torch.no_grad():
        x = vision_model.patch_embedding_3d(volume_3d)
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = vision_model.get_position_embedding_3d(Dp, Hp * Wp, Hp, Wp).to(x.device, dtype=x.dtype)
        x = x + pos
        x = vision_model.encoder(x).last_hidden_state
        x = vision_model.post_layernorm(x)

    x = x.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, model.num_vision_tokens)
    x = x.transpose(1, 2)

    x = model.vision_projector(x.to(torch.bfloat16))
    x = x * model.vis_scale.to(x.dtype)
    return x


def image_embed_from_alignment(model: BrainGemma3D, volumes, use_phase1_forward=True):
    """use Phase 1 forward (trainable patch_embedding) or Phase 2 (no_grad ViT)"""
    if isinstance(volumes, (list, tuple)):
        volumes = torch.cat(volumes, dim=0)
    
    if use_phase1_forward:
        vis_tokens = encode_volume_for_alignment_phase1(model, volumes)
    else:
        vis_tokens = encode_volume_for_alignment_no_grad_vit(model, volumes)
    
    v = vis_tokens.mean(dim=1) if vis_tokens.shape[1] > 1 else vis_tokens.squeeze(1)
    return F.normalize(v.float(), p=2, dim=-1)


def clip_infonce_loss(v, t, temperature=0.07):
    """Bidirectional contrastive loss"""
    logits = (v @ t.T) / temperature
    targets = torch.arange(v.size(0), device=v.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2


def get_alignment_params_grouped(model: BrainGemma3D, lr_patch=1e-4, lr_proj=5e-4, lr_scale=1e-3):
    """
    Return param groups with differentiated LR for Phase 1.
    """
    param_groups = []
    
    patch_params = [p for p in model.vision_encoder.vision_model.patch_embedding_3d.parameters() if p.requires_grad]
    if patch_params:
        param_groups.append({"params": patch_params, "lr": lr_patch})
    
    proj_params = [p for p in model.vision_projector.parameters() if p.requires_grad]
    if proj_params:
        param_groups.append({"params": proj_params, "lr": lr_proj})
    
    if hasattr(model, "vis_scale") and model.vis_scale.requires_grad:
        param_groups.append({"params": [model.vis_scale], "lr": lr_scale})
    
    return param_groups


def run_phase1_alignment_advanced(
    model: BrainGemma3D,
    dataset: List[Dict],
    epochs=5,
    batch_size=1,
    lr=2e-4,
    weight_decay=0.01,
    temperature=0.07,
    max_text_len=128,
    seed=0,
    train_frac=0.9,
    max_train=None,
    max_val=32,
    target_size=(64, 128, 128),
    val_every=1,
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,
):
    """
    Phase 1: Contrastive alignment vision-report (NO prompt).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    data = dataset.copy()
    random.shuffle(data)
    if max_train is not None:
        data = data[:max_train]

    n_train = max(2, int(len(data) * train_frac))
    train_data = data[:n_train]
    val_data = data[n_train:n_train + max_val] if (len(data) > n_train) else []

    # Freeze LM
    model.language_model.eval()
    for p in model.language_model.parameters():
        p.requires_grad = False

    # Vision encoder: ONLY patch_embedding_3d trainable
    model.vision_encoder.eval()
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    
    for p in model.vision_encoder.vision_model.patch_embedding_3d.parameters():
        p.requires_grad = True
    print("✅ Phase 1: patch_embedding_3d UNLOCKED (trainable)")

    # Projector train
    model.vision_projector.train()
    
    if hasattr(model, "vis_scale"):
        model.vis_scale.requires_grad = True

    param_groups = get_alignment_params_grouped(
        model, 
        lr_patch=lr * 0.2,
        lr_proj=lr,
        lr_scale=lr * 2.0
    )
    optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    best_val = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    print("=" * 70)
    print("🔧 PHASE 1 - IMAGE-TEXT GROUNDING (NO PROMPT)")
    print("=" * 70)
    print(f"Total={len(data)} | Train={len(train_data)} | Val={len(val_data)}")
    print(f"Epochs={epochs} | Batch={batch_size} | LR={lr} | Temp={temperature}")
    print(f"Early Stopping: patience={early_stopping_patience}")
    print(f"Healthy samples: {sum(1 for ex in train_data if ex.get('is_healthy', False))}/{len(train_data)}")

    def _iter_batches(split):
        for i in range(0, len(split), batch_size):
            yield split[i:i+batch_size]

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in _iter_batches(train_data):
            vols = []
            texts = []
            for ex in batch:
                vol = get_volume_from_ex(ex, target_size, device=model.lm_device)
                vols.append(vol)
                texts.append(ex["report"])
            
            vols = torch.cat(vols, dim=0)
            
            v_emb = image_embed_from_alignment(model, vols, use_phase1_forward=True)
            t_emb = text_embed_from_lm(model, texts, max_length=max_text_len)
            
            loss = clip_infonce_loss(v_emb, t_emb, temperature=temperature)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            train_loss_sum += loss.item()
            train_count += 1

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # Validation
        val_loss_avg = 0.0
        if val_data and (ep % val_every == 0):
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in _iter_batches(val_data):
                    vols = []
                    texts = []
                    for ex in batch:
                        vol = get_volume_from_ex(ex, target_size, device=model.lm_device)
                        vols.append(vol)
                        texts.append(ex["report"])
                    
                    vols = torch.cat(vols, dim=0)
                    v_emb = image_embed_from_alignment(model, vols, use_phase1_forward=True)
                    t_emb = text_embed_from_lm(model, texts, max_length=max_text_len)
                    loss = clip_infonce_loss(v_emb, t_emb, temperature=temperature)
                    val_loss_sum += loss.item()
                    val_count += 1
            val_loss_avg = val_loss_sum / max(val_count, 1)
            
            # Early stopping
            if val_loss_avg < best_val - early_stopping_min_delta:
                best_val = val_loss_avg
                best_epoch = ep
                patience_counter = 0
                best_model_state = save_best_model_state(model)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"⏹️  Early stopping at epoch {ep}")
                break

        print(f"Epoch {ep}/{epochs} | Train={train_loss_avg:.4f} | Val={val_loss_avg:.4f} | vis_scale={model.vis_scale.item():.3f}")

    if best_model_state is not None:
        load_best_model_state(model, best_model_state)
    
    print(f"✅ Phase 1 (alignment) completed | Best Val={best_val:.4f} at epoch {best_epoch}")
    print(f"📊 Final vis_scale: {model.vis_scale.item():.3f}")
    
    # Re-freeze patch_embedding_3d for Phase 2
    for p in model.vision_encoder.vision_model.patch_embedding_3d.parameters():
        p.requires_grad = False
    print("🔒 patch_embedding_3d RE-FROZEN for Phase 2")
    
    return


# ============================================================
# PHASE 2A - SUPERVISED (projector only)
# ============================================================

def build_inputs_and_labels(model: BrainGemma3D, ex: Dict, prompt: str, max_text_len=128, target_size=(64, 128, 128)):
    """Build inputs_embeds and labels for Phase 2 training"""
    vol = get_volume_from_ex(ex, target_size).to(model.lm_device, dtype=torch.bfloat16)

    vis = encode_volume_for_alignment_no_grad_vit(model, vol)

    tok_prompt = model.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=max_text_len
    ).to(model.lm_device)

    tok_target = model.tokenizer(
        ex["report"], return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_text_len
    ).to(model.lm_device)

    prompt_ids = tok_prompt.input_ids
    target_ids = tok_target.input_ids
    
    # MANUAL EOS ADDITION
    eos_token_id = model.tokenizer.eos_token_id
    if target_ids[0, -1] != eos_token_id:
        eos_tensor = torch.tensor([[eos_token_id]], device=model.lm_device)
        target_ids = torch.cat([target_ids, eos_tensor], dim=1)
    
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)

    text_embeds = model.language_model.get_input_embeddings()(full_ids)

    inputs_embeds = torch.cat([vis, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)

    K = vis.shape[1]
    labels = torch.cat([
        torch.full((1, K), -100, device=inputs_embeds.device, dtype=torch.long),
        torch.full_like(prompt_ids, -100),
        target_ids
    ], dim=1)

    return inputs_embeds, attention_mask, labels


def set_trainable_phase2A(model: BrainGemma3D):
    """Configure trainability for Phase 2A"""
    for p in model.language_model.parameters():
        p.requires_grad = False

    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    model.vision_encoder.eval()

    for p in model.vision_projector.parameters():
        p.requires_grad = True
    model.vision_projector.train()

    if hasattr(model, "vis_scale"):
        model.vis_scale.requires_grad = True


def save_best_model_state(model: BrainGemma3D) -> Dict:
    """Save trainable state of the model in memory"""
    state = {
        "vision_projector": {k: v.cpu().clone() for k, v in model.vision_projector.state_dict().items()},
    }
    
    if hasattr(model, "vis_scale"):
        state["vis_scale"] = model.vis_scale.data.cpu().clone()
    
    lora_params = {k: v.cpu().clone() for k, v in model.language_model.state_dict().items() 
                   if "lora" in k.lower()}
    if lora_params:
        state["lora_params"] = lora_params
    
    return state


def load_best_model_state(model: BrainGemma3D, state: Dict):
    """Restore the saved best model state"""
    model.vision_projector.load_state_dict({k: v.to(model.lm_device) 
                                           for k, v in state["vision_projector"].items()})
    
    if "vis_scale" in state and hasattr(model, "vis_scale"):
        model.vis_scale.data = state["vis_scale"].to(model.lm_device)
    
    if "lora_params" in state:
        current_state = model.language_model.state_dict()
        current_state.update({k: v.to(model.lm_device) for k, v in state["lora_params"].items()})
        model.language_model.load_state_dict(current_state, strict=False)
    
    print("✅ Best model state restored")


def run_phase2A_advanced(
    model: BrainGemma3D,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    epochs=3,
    lr=2e-4,
    weight_decay=0.01,
    max_text_len=192,
    batch_size=1,
    grad_accum=8,
    target_size=(64, 128, 128),
    val_every=1,
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,
    seed=0,
):
    """Phase 2A - Report Generation (short) with CANONICAL PROMPT"""
    
    set_trainable_phase2A(model)

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    model.language_model.config.use_cache = False

    params = list(model.vision_projector.parameters())
    if hasattr(model, "vis_scale") and model.vis_scale.requires_grad:
        params.append(model.vis_scale)

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    print("=" * 70)
    print("🧠 PHASE 2A - Report Generation (short)")
    print("=" * 70)
    print(f"epochs={epochs} | batch_size={batch_size} | grad_accum={grad_accum}")
    print(f"lr={lr} | max_text_len={max_text_len} tokens")
    print(f"Prompt: CANONICAL - '{CANONICAL_PROMPT}'")
    print(f"Early Stopping: patience={early_stopping_patience}")
    print(f"Healthy samples: {sum(1 for ex in train_data if ex.get('is_healthy', False))}/{len(train_data)}")
    
    steps_per_epoch = len(train_data)

    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        
        train_loss_sum = 0.0
        train_count = 0
        
        for i, ex in enumerate(train_data):
            try:
                inputs_embeds, attention_mask, labels = build_inputs_and_labels(
                    model, ex, CANONICAL_PROMPT, max_text_len, target_size
                )
                
                out = model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                
                loss = out.loss / grad_accum
                loss.backward()
                
                train_loss_sum += out.loss.item()
                train_count += 1
                
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_data):
                    optim.step()
                    optim.zero_grad()
                    
            except Exception as e:
                print(f"Error training batch {i}: {e}")
                optim.zero_grad()
                continue

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # Validation
        val_loss_avg = 0.0
        if val_data and (ep % val_every == 0):
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for ex in val_data[:32]:
                    try:
                        inputs_embeds, attention_mask, labels = build_inputs_and_labels(
                            model, ex, CANONICAL_PROMPT, max_text_len, target_size
                        )
                        out = model.language_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True,
                        )
                        val_loss_sum += out.loss.item()
                        val_count += 1
                    except:
                        continue
            val_loss_avg = val_loss_sum / max(val_count, 1)
            
            # Early stopping
            if val_loss_avg < best_val - early_stopping_min_delta:
                best_val = val_loss_avg
                best_epoch = ep
                patience_counter = 0
                best_model_state = save_best_model_state(model)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"⏹️  Early stopping at epoch {ep}")
                break

        print(f"Epoch {ep}/{epochs} | Train={train_loss_avg:.4f} | Val={val_loss_avg:.4f}")

    if best_model_state is not None:
        load_best_model_state(model, best_model_state)
    
    print(f"✅ Phase 2A completed | Best Val={best_val:.4f} at epoch {best_epoch}")
    return model


# ============================================================
# PHASE 2B - SUPERVISED (projector + LoRA)
# ============================================================

def add_lora_for_phase2B(model: BrainGemma3D, r=4, alpha=8, dropout=0.05):
    """Add LoRA adapters to the language model"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.print_trainable_parameters()


def set_trainable_phase2B(model: BrainGemma3D):
    """Configure trainability for Phase 2B"""
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    model.vision_encoder.eval()

    for p in model.vision_projector.parameters():
        p.requires_grad = True
    model.vision_projector.train()

    if hasattr(model, "vis_scale"):
        model.vis_scale.requires_grad = True


def run_phase2B_advanced(
    model: BrainGemma3D,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    epochs=3,
    lr_proj=2e-4,
    lr_lora=5e-5,
    weight_decay=0.01,
    batch_size=1,
    grad_accum=8,
    max_text_len=256,
    target_size=(64, 128, 128),
    val_every=1,
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,
    seed=0,
):
    """Phase 2B - Report Generation (full) with CANONICAL PROMPT"""
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    model.language_model.config.use_cache = False

    set_trainable_phase2B(model)

    proj_params = list(model.vision_projector.parameters())
    if hasattr(model, "vis_scale"):
        proj_params.append(model.vis_scale)

    lora_params = [p for p in model.language_model.parameters() if p.requires_grad]

    optim = torch.optim.AdamW([
        {"params": proj_params, "lr": lr_proj, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay},
    ])

    best_val = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    print("=" * 70)
    print("🧠 PHASE 2B - Report Generation (full)")
    print("=" * 70)
    print(f"epochs={epochs} | batch_size={batch_size} | grad_accum={grad_accum}")
    print(f"lr_proj={lr_proj} | lr_lora={lr_lora} | max_text_len={max_text_len} tokens")
    print(f"Prompt: CANONICAL - '{CANONICAL_PROMPT}'")
    print(f"Early Stopping: patience={early_stopping_patience}")
    print(f"Healthy samples: {sum(1 for ex in train_data if ex.get('is_healthy', False))}/{len(train_data)}")
    
    steps_per_epoch = len(train_data)

    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        
        train_loss_sum = 0.0
        train_count = 0
        
        for i, ex in enumerate(train_data):
            try:
                inputs_embeds, attention_mask, labels = build_inputs_and_labels(
                    model, ex, CANONICAL_PROMPT, max_text_len, target_size
                )
                
                out = model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                
                loss = out.loss / grad_accum
                loss.backward()
                
                train_loss_sum += out.loss.item()
                train_count += 1
                
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_data):
                    optim.step()
                    optim.zero_grad()
                    
            except Exception as e:
                print(f"Error training batch {i}: {e}")
                optim.zero_grad()
                continue

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # Validation
        val_loss_avg = 0.0
        if val_data and (ep % val_every == 0):
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for ex in val_data[:32]:
                    try:
                        inputs_embeds, attention_mask, labels = build_inputs_and_labels(
                            model, ex, CANONICAL_PROMPT, max_text_len, target_size
                        )
                        out = model.language_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True,
                        )
                        val_loss_sum += out.loss.item()
                        val_count += 1
                    except:
                        continue
            val_loss_avg = val_loss_sum / max(val_count, 1)
            
            # Early stopping
            if val_loss_avg < best_val - early_stopping_min_delta:
                best_val = val_loss_avg
                best_epoch = ep
                patience_counter = 0
                best_model_state = save_best_model_state(model)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"⏹️  Early stopping at epoch {ep}")
                break

        print(f"Epoch {ep}/{epochs} | Train={train_loss_avg:.4f} | Val={val_loss_avg:.4f}")

    if best_model_state is not None:
        load_best_model_state(model, best_model_state)
    
    print(f"✅ Phase 2B completed | Best Val={best_val:.4f} at epoch {best_epoch}")
    return model


# ============================================================
# UTILITIES
# ============================================================

def save_full_package(model: BrainGemma3D, out_dir="ckpt_full"):
    """Save projector + vis_scale + lora"""
    os.makedirs(out_dir, exist_ok=True)
    
    state = {
        "vision_projector": model.vision_projector.state_dict(),
        "vis_scale": model.vis_scale.item() if hasattr(model, "vis_scale") else None,
    }
    torch.save(state, os.path.join(out_dir, "projector_vis_scale.pt"))
    
    # Save LoRA if present
    if hasattr(model.language_model, "save_pretrained"):
        lora_dir = os.path.join(out_dir, "lora_adapters")
        model.language_model.save_pretrained(lora_dir)
        print(f"✅ Saved LoRA adapters to {lora_dir}")
    
    print(f"💾 Saved full package to {out_dir}")


def save_volume_slices(vol: torch.Tensor, save_path: str, title: str = "Volume", num_slices: int = 5, is_healthy: bool = False):
    """
    Save ALL slices of the preprocessed 3D volume as a complete grid.
    Shows exactly what is passed to the model after load_nifti_volume().
    
    IMPORTANT: use origin="upper" to match preprocessing!
    vol: (1,1,D,H,W) or (D,H,W)
    is_healthy: bool to determine image orientation (origin)
    """
    if vol.ndim == 5:
        vol = vol.squeeze(0).squeeze(0)  # (D,H,W)
    elif vol.ndim == 4:
        vol = vol.squeeze(0)  # (D,H,W)
    
    vol_np = vol.cpu().numpy()
    D, H, W = vol_np.shape
    
    # SHOW ALL SLICES in a grid
    n_cols = 8
    n_rows = (D + n_cols - 1) // n_cols  # ceil division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), facecolor="black")
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Global range for consistent normalization
    vmin, vmax = vol_np.min(), vol_np.max()
    
    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < D:
            slice_img = vol_np[idx]
            slice_mean = slice_img.mean()
            slice_std = slice_img.std()
            
            # BraTS: lower (after transpose)
            # HealthyBrains: upper (preprocessed)
            origin = "lower"
            
            ax.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax, origin=origin)
            ax.set_title(f'z={idx}\nμ={slice_mean:.3f}', color='cyan', fontsize=8, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')  # Hide empty cells
    
    info_text = f"Shape: {vol_np.shape} | Range: [{vmin:.3f}, {vmax:.3f}] | Mean: {vol_np.mean():.3f} | Std: {vol_np.std():.3f}"
    fig.suptitle(f"{title}\n{info_text}", fontsize=10, fontweight='bold', color='white')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"💾 Saved ALL {D} slices: {save_path}")


# ============================================================
# DIAGNOSTIC UTILITIES
# ============================================================

@torch.no_grad()
def make_variants_from_volume(vol: torch.Tensor, add_noise: bool = True) -> Dict[str, torch.Tensor]:
    """
    vol: (1,1,D,H,W) float
    Returns variants:
      - real: original vol
      - zero: all zeros
      - shuf: voxels shuffled (same distribution, different structure)
      - noise: gaussian noise (optional)
    """
    assert vol.ndim == 5 and vol.shape[0] == 1 and vol.shape[1] == 1, f"Unexpected shape: {vol.shape}"

    real = vol.clone()
    zero = torch.zeros_like(vol)

    flat = vol.flatten()
    perm = torch.randperm(flat.numel(), device=flat.device)
    shuf = flat[perm].view_as(vol)

    out = {"real": real, "zero": zero, "shuf": shuf}

    if add_noise:
        # noise with std similar to vol (if vol in [0,1], std ~ 0.2 is ok)
        noise = torch.randn_like(vol) * 0.2
        # clamp to keep it within a reasonable range
        noise = noise.clamp(-1.0, 1.0)
        out["noise"] = noise

    return out


@torch.no_grad()
def summarize_tokens(name: str, tokens: torch.Tensor) -> str:
    """
    tokens: (B,K,H) (here B=1)
    Returns a string with token statistics
    """
    t = tokens.float()
    msg = (
        f"{name}: shape={tuple(tokens.shape)} | "
        f"mean={t.mean().item():.4f} std={t.std().item():.4f} "
        f"min={t.min().item():.4f} max={t.max().item():.4f} "
        f"norm={t.norm(dim=-1).mean().item():.4f}"
    )
    return msg


@torch.no_grad()
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors"""
    a = a.float().view(-1)
    b = b.float().view(-1)
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum().item())


@torch.no_grad()
def run_real_zero_shuf_diagnostics(
    model,
    ex: Dict,
    prompt: str = None,
    target_size=(64, 128, 128),
    max_new_tokens: int = 128,
    min_new_tokens: int = 10,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    add_noise: bool = True,
    save_dir: str = "diagnostics_output",
):
    """
    Run full diagnostics:
    - load REAL volume
    - create ZERO / SHUF (+ NOISE)
    - compare vision tokens
    - generate report for each case
    - SAVE the images of the variants
    """
    if prompt is None:
        prompt = CANONICAL_PROMPT
        
    model.eval()

    # volume CPU -> then encode_volume will move it to lm_device
    vol_real = load_nifti_volume(ex["image_path"], target_size=target_size)  # (1,1,D,H,W) float CPU

    variants = make_variants_from_volume(vol_real, add_noise=add_noise)

    print("\n" + "=" * 70)
    print("🔬 DIAGNOSTICS: REAL vs ZERO vs SHUF" + (" (+ NOISE)" if add_noise else ""))
    print("=" * 70)
    print(f"patient_id={ex.get('patient_id', 'N/A')} | image_path={ex['image_path']}")
    
    # Save the variants as images
    os.makedirs(save_dir, exist_ok=True)
    patient_id = ex.get('patient_id', 'unknown')
    is_healthy_data = ex.get('is_healthy', False) or "HealthyBrains" in ex.get("image_path", "") or "healthy" in ex.get("image_path", "").lower()
    for variant_name, variant_vol in variants.items():
        save_path = os.path.join(save_dir, f"{patient_id}_{variant_name}.png")
        save_volume_slices(variant_vol, save_path, title=f"{patient_id} - {variant_name.upper()}", is_healthy=is_healthy_data)

    # ---- Vision tokens
    tokens = {}
    for k, v in variants.items():
        tokens[k] = model.encode_volume(v)  # (1,K,H)

    for k in tokens:
        print("  " + summarize_tokens(k, tokens[k]))

    # cosine similarities (flattened)
    keys = list(tokens.keys())
    base = "real"
    for k in keys:
        if k == base:
            continue
        cs = cosine(tokens[base], tokens[k])
        print(f"  cosine({base},{k}) = {cs:.4f}")

    # ---- Generate reports
    print("\n" + "-" * 70)
    print("📝 Generated reports (same prompt, different inputs)")
    print("-" * 70)

    for k, v in variants.items():
        rep = model.generate_report(
            v,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        print(f"\n[{k.upper()}]\n{rep}\n")


@torch.no_grad()
def generate_sample_reports(
    model,
    dataset: List[Dict],
    n_samples: int = 3,
    prompt: str = None,
    target_size=(64, 128, 128),
    max_new_tokens: int = 128,
    min_new_tokens: int = 10,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    seed: int = 0,
    save_dir: str = "sample_reports_output",
):
    """
    Select n_samples examples from the dataset and generate reports.
    Also prints the first lines of the GT report for comparison.
    Also SAVES the volume images.
    
    If prompt=None, uses CANONICAL_PROMPT
    """
    # Use CANONICAL_PROMPT if none provided
    if prompt is None:
        prompt = CANONICAL_PROMPT
        
    rng = random.Random(seed)
    model.eval()

    picks = dataset if len(dataset) <= n_samples else rng.sample(dataset, n_samples)

    print("\n" + "=" * 70)
    print(f"🧪 SAMPLE REPORT GENERATION (n={len(picks)})")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)

    for i, ex in enumerate(picks, 1):
        vol = load_nifti_volume(ex["image_path"], target_size=target_size)
        
        # Save the volume as image
        patient_id = ex.get('patient_id', f'sample_{i}')
        is_healthy_data = ex.get('is_healthy', False) or "HealthyBrains" in ex.get("image_path", "") or "healthy" in ex.get("image_path", "").lower()
        save_path = os.path.join(save_dir, f"{patient_id}_input.png")
        save_volume_slices(vol, save_path, title=f"{patient_id} - INPUT", is_healthy=is_healthy_data)

        gen = model.generate_report(
            vol,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        gt = ex.get("report", "")
        gt_preview = (gt[:400] + ("..." if len(gt) > 400 else ""))

        print("\n" + "-" * 70)
        print(f"[{i}] patient_id={ex.get('patient_id','N/A')}")
        print(f"GT (preview):\n{gt_preview}")
        print("\nGEN:\n" + gen)


def generate_experiment_name(args) -> str:
    """Generate automatic experiment name from hyperparameters"""
    tokens = f"tok{args.num_vision_tokens}"
    lr1 = f"lr1_{args.phase1_lr:.0e}".replace("e-0", "e-")
    lr2a = f"lr2a_{args.phase2a_lr:.0e}".replace("e-0", "e-")
    lr2b_proj = f"lr2bp_{args.phase2b_lr_proj:.0e}".replace("e-0", "e-")
    lr2b_lora = f"lr2bl_{args.phase2b_lr_lora:.0e}".replace("e-0", "e-")
    acc = f"acc{args.phase2a_grad_accum}"
    epochs = f"ep{args.phase1_epochs}-{args.phase2a_epochs}-{args.phase2b_epochs}"
    strat = "strat" if not args.no_stratify_lesion_side else "nostrat"
    
    name = f"medgemma3d_{tokens}_{lr1}_{lr2a}_{lr2b_proj}_{lr2b_lora}_{acc}_{epochs}_{strat}"
    
    return name


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BrainGemma3D Advanced Training")
    
    # Paths
    parser.add_argument("--base-dir", type=str, default="/leonardo_work/CESMA_leonardo/CBMS")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints/exp_name/phase2a_projector)")
    
    # Dataset
    parser.add_argument("--num-brats-patients", type=int, default=None)
    parser.add_argument("--num-healthy-patients", type=int, default=None)
    parser.add_argument("--modality", type=str, default="flair")
    parser.add_argument("--target-size", type=int, nargs=3, default=[64, 128, 128])
    parser.add_argument("--no-stratify-lesion-side", action="store_true", help="Disable stratification by lesion side")
    
    # Model
    parser.add_argument("--num-vision-tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    
    # Phase 1
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase1-batch", type=int, default=1)
    parser.add_argument("--phase1-lr", type=float, default=5e-4)
    parser.add_argument("--skip-phase1", action="store_true")
    
    # Phase 2A
    parser.add_argument("--phase2a-epochs", type=int, default=10)
    parser.add_argument("--phase2a-lr", type=float, default=2e-4)
    parser.add_argument("--phase2a-grad-accum", type=int, default=8)
    parser.add_argument("--skip-phase2a", action="store_true")
    
    # Phase 2B
    parser.add_argument("--phase2b-epochs", type=int, default=15)
    parser.add_argument("--phase2b-lr-proj", type=float, default=2e-4)
    parser.add_argument("--phase2b-lr-lora", type=float, default=5e-5)
    parser.add_argument("--phase2b-grad-accum", type=int, default=8)
    parser.add_argument("--skip-phase2b", action="store_true")
    
    # Training features
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    
    # Generation
    parser.add_argument("--gen-max-new", type=int, default=256)
    parser.add_argument("--gen-min-new", type=int, default=10, help="Minimum tokens to generate (prevents empty outputs)")
    parser.add_argument("--gen-temperature", type=float, default=0.1)
    parser.add_argument("--gen-top-p", type=float, default=0.9)
    parser.add_argument("--gen-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--gen-no-repeat-ngram", type=int, default=3)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gen-samples", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip diagnostics (real/zero/shuf) after training")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        experiment_name = generate_experiment_name(args)
    else:
        experiment_name = args.experiment_name
    
    # Paths
    base_dir = Path(args.base_dir)
    vision_model_path = str(base_dir / "Models" / "siglip")
    language_model_path = str(base_dir / "Models" / "medgemma")
    brats_images = str(base_dir / "Datasets" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData")
    brats_reports = str(base_dir / "Datasets" / "TextBraTS" / "TextBraTSData")
    healthy_brains = str(base_dir / "Datasets" / "HealthyBrains_Preprocessed")
    
    checkpoint_base = os.path.join(args.checkpoint_dir, experiment_name)
    output_base = os.path.join(args.output_dir, experiment_name)
    os.makedirs(checkpoint_base, exist_ok=True)
    os.makedirs(output_base, exist_ok=True)
    
    TARGET_SIZE = tuple(args.target_size)
    
    print("\n" + "="*70)
    print("BRAINGEMMA3D - TRAINING PIPELINE")
    print("="*70)
    print(f"📂 Experiment: {experiment_name}")
    print(f"📂 Checkpoints: {checkpoint_base}/")
    print(f"📂 Outputs: {output_base}/")
    print("="*70 + "\n")
    
    # Load dataset
    dataset = build_balanced_dataset(
        brats_images_base=brats_images,
        brats_reports_base=brats_reports,
        healthy_brains_base=healthy_brains,
        num_brats_patients=args.num_brats_patients,
        num_healthy_patients=args.num_healthy_patients,
        modality=args.modality,
    )
    
    if len(dataset) < 4:
        raise ValueError("Dataset too small for meaningful split")
    
    print(f"\n✅ Dataset loaded: {len(dataset)} total examples")
    
    # Split with stratification by lesion_side
    train_data, val_data, test_data = make_group_split(
        dataset, 
        seed=args.seed, 
        group_key="image_path",
        stratify_by_lesion_side=not args.no_stratify_lesion_side,
    )
    
    # Initialize model
    print(f"\n🤖 Initializing BrainGemma3D model...")
    model = BrainGemma3D(
        vision_model_dir=vision_model_path,
        language_model_dir=language_model_path,
        depth=args.depth,
        num_vision_tokens=args.num_vision_tokens,
        freeze_vision=False,
        freeze_language=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    
    print(f"✅ Model initialized on device: {model.lm_device}")
    
    # -------------------------
    # LOAD CHECKPOINT (if specified)
    # -------------------------
    if args.load_checkpoint:
        print(f"\n📥 Loading checkpoint from: {args.load_checkpoint}")
        
        # Load projector + vis_scale
        proj_path = os.path.join(args.load_checkpoint, "projector_vis_scale.pt")
        if os.path.exists(proj_path):
            ckpt = torch.load(proj_path, map_location=model.lm_device)
            model.vision_projector.load_state_dict(ckpt["vision_projector"])
            if "vis_scale" in ckpt and ckpt["vis_scale"] is not None:
                val = ckpt["vis_scale"]
                if isinstance(val, torch.Tensor):
                    model.vis_scale.data = val.to(model.lm_device)
                else:
                    model.vis_scale.data.fill_(val)
            print(f"  ✅ Loaded projector | vis_scale={model.vis_scale.item():.3f}")
        else:
            print(f"  ⚠️  Projector checkpoint not found at {proj_path}")
        
        # Load LoRA adapters (if exists - for resuming from Phase 2B)
        lora_dir = os.path.join(args.load_checkpoint, "lora_adapters")
        if os.path.exists(lora_dir):
            try:
                from peft import PeftModel
                model.language_model = PeftModel.from_pretrained(
                    model.language_model, 
                    lora_dir,
                    is_trainable=True
                )
                print(f"  ✅ Loaded LoRA adapters from {lora_dir}")
            except Exception as e:
                print(f"  ⚠️  Could not load LoRA adapters: {e}")
        else:
            print(f"  ℹ️  No LoRA adapters found (resuming from Phase 1 or 2A)")
        
        print(f"✅ Checkpoint loaded successfully!\n")
    
    # PHASE 1
    if not args.skip_phase1:
        print("\n" + "="*70)
        print("PHASE 1: ALIGNMENT")
        print("="*70)
        run_phase1_alignment_advanced(
            model,
            train_data,
            epochs=args.phase1_epochs,
            batch_size=args.phase1_batch,
            lr=args.phase1_lr,
            target_size=TARGET_SIZE,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
        
        phase1_dir = os.path.join(checkpoint_base, "phase1_alignment")
        save_full_package(model, phase1_dir)
        
        # Diagnostics after Phase 1
        if not args.skip_diagnostics:
            print("\n🔬 Running Phase 1 diagnostics...")
            diagnostics_dir = os.path.join(output_base, "phase1_diagnostics")
            samples_dir = os.path.join(output_base, "phase1_samples")
            os.makedirs(diagnostics_dir, exist_ok=True)
            os.makedirs(samples_dir, exist_ok=True)
            
            if len(train_data) > 0:
                generate_sample_reports(
                    model,
                    dataset=train_data,
                    n_samples=min(2, len(train_data)),
                    target_size=TARGET_SIZE,
                    max_new_tokens=args.gen_max_new,
                    min_new_tokens=args.gen_min_new,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram,
                    seed=args.seed,
                    save_dir=samples_dir,
                )
                
                run_real_zero_shuf_diagnostics(
                    model,
                    ex=train_data[0],
                    target_size=TARGET_SIZE,
                    max_new_tokens=min(128, args.gen_max_new),
                    min_new_tokens=args.gen_min_new,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram,
                    add_noise=True,
                    save_dir=diagnostics_dir,
                )
    
    # PHASE 2A
    if not args.skip_phase2a:
        print("\n" + "="*70)
        print("PHASE 2A: PROJECTOR FINE-TUNING")
        print("="*70)
        model = run_phase2A_advanced(
            model,
            train_data,
            val_data,
            epochs=args.phase2a_epochs,
            lr=args.phase2a_lr,
            grad_accum=args.phase2a_grad_accum,
            target_size=TARGET_SIZE,
            early_stopping_patience=args.early_stopping_patience,
        )
        
        phase2a_dir = os.path.join(checkpoint_base, "phase2a_projector")
        save_full_package(model, phase2a_dir)
        
        # Diagnostics after Phase 2A
        if not args.skip_diagnostics:
            print("\n🔬 Running Phase 2A diagnostics...")
            diagnostics_dir = os.path.join(output_base, "phase2a_diagnostics")
            samples_dir = os.path.join(output_base, "phase2a_samples")
            os.makedirs(diagnostics_dir, exist_ok=True)
            os.makedirs(samples_dir, exist_ok=True)
            
            eval_data = val_data if len(val_data) > 0 else train_data
            if len(eval_data) > 0:
                generate_sample_reports(
                    model,
                    dataset=eval_data,
                    n_samples=min(2, len(eval_data)),
                    target_size=TARGET_SIZE,
                    max_new_tokens=args.gen_max_new,
                    min_new_tokens=args.gen_min_new,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram,
                    seed=args.seed + 1,
                    save_dir=samples_dir,
                )
                
                run_real_zero_shuf_diagnostics(
                    model,
                    ex=eval_data[0],
                    target_size=TARGET_SIZE,
                    max_new_tokens=min(128, args.gen_max_new),
                    min_new_tokens=args.gen_min_new,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram,
                    add_noise=True,
                    save_dir=diagnostics_dir,
                )
    
    # PHASE 2B
    if not args.skip_phase2b:
        print("\n" + "="*70)
        print("PHASE 2B: FULL FINE-TUNING (PROJECTOR + LoRA)")
        print("="*70)
        add_lora_for_phase2B(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        
        model = run_phase2B_advanced(
            model,
            train_data,
            val_data,
            epochs=args.phase2b_epochs,
            lr_proj=args.phase2b_lr_proj,
            lr_lora=args.phase2b_lr_lora,
            grad_accum=args.phase2b_grad_accum,
            max_text_len=256,
            target_size=TARGET_SIZE,
            early_stopping_patience=args.early_stopping_patience,
        )
        
        phase2b_dir = os.path.join(checkpoint_base, "phase2b_final")
        save_full_package(model, phase2b_dir)
        
        # Diagnostics after Phase 2B
        if not args.skip_diagnostics:
            print("\n🔬 Running Phase 2B diagnostics...")
            diagnostics_dir = os.path.join(output_base, "phase2b_diagnostics")
            samples_dir = os.path.join(output_base, "phase2b_samples")
            os.makedirs(diagnostics_dir, exist_ok=True)
            os.makedirs(samples_dir, exist_ok=True)
            
            eval_data = val_data if len(val_data) > 0 else train_data
            if len(eval_data) > 0:
                generate_sample_reports(
                    model,
                    dataset=eval_data,
                    n_samples=min(3, len(eval_data)),
                    target_size=TARGET_SIZE,
                    max_new_tokens=args.gen_max_new,
                    min_new_tokens=args.gen_min_new,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram,
                    seed=args.seed + 2,
                    save_dir=samples_dir,
                )
                
                run_real_zero_shuf_diagnostics(
                    model,
                    ex=eval_data[0],
                    target_size=TARGET_SIZE,
                    max_new_tokens=args.gen_max_new,
                    min_new_tokens=args.gen_min_new,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram,
                    add_noise=True,
                    save_dir=diagnostics_dir,
                )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    
    # Final test set evaluation
    if len(test_data) > 0:
        if not args.skip_gen_samples:
            print("\n📝 Generating final test set reports...")
            final_samples_dir = os.path.join(output_base, "final_test_samples")
            os.makedirs(final_samples_dir, exist_ok=True)
            
            generate_sample_reports(
                model,
                dataset=test_data,
                n_samples=min(5, len(test_data)),
                target_size=TARGET_SIZE,
                max_new_tokens=args.gen_max_new,
                min_new_tokens=args.gen_min_new,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                repetition_penalty=args.gen_repetition_penalty,
                no_repeat_ngram_size=args.gen_no_repeat_ngram,
                seed=args.seed + 999,
                save_dir=final_samples_dir,
            )
        
        if not args.skip_diagnostics:
            print("\n🔬 Running final test diagnostics...")
            final_diagnostics_dir = os.path.join(output_base, "final_test_diagnostics")
            os.makedirs(final_diagnostics_dir, exist_ok=True)
            
            run_real_zero_shuf_diagnostics(
                model,
                ex=test_data[0],
                target_size=TARGET_SIZE,
                max_new_tokens=args.gen_max_new,
                min_new_tokens=args.gen_min_new,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                repetition_penalty=args.gen_repetition_penalty,
                no_repeat_ngram_size=args.gen_no_repeat_ngram,
                add_noise=True,
                save_dir=final_diagnostics_dir,
            )
    
    print("\n🎉 ALL DONE!")
    print(f"\n💡 Use the Phase 2B checkpoint for inference:")
    print(f"   python braingemma3d_inference.py --checkpoint-dir {os.path.join(checkpoint_base, 'phase2b_final')}")


if __name__ == "__main__":
    main()
