#!/usr/bin/env python3
"""
M3D-LaMed Baseline Script for BraTS Report Generation
======================================================
This script adapts the M3D-LaMed model (GoodBaiBai88/M3D-LaMed-Phi-3-4B) to 
generate radiology reports from BraTS NIfTI files.

Combines:
- Input loading from braingemma3d_architecture.py (NIfTI → 3D volume)
- Prompt structure from braingemma3d_pure.py (CANONICAL_PROMPT)
- M3D-LaMed quickstart API

M3D-LaMed features:
- Input shape: 1 × 64 × 128 × 128 (batch, depth, height, width)
- Normalization: 0-1 (Min-Max)
- Format: .npy array (but we support conversion from NIfTI)
- Projection tokens: 256 <im_patch> tokens

Usage:
    python m3d_baseline.py --nifti_path path/to/scan.nii.gz --output_dir results/
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

# ============================================================================
# PROMPT CONFIGURATION - CANONICAL
# ============================================================================

# System instruction to guide the model (neuroradiology MRI-specific)
INSTRUCTION = (
    "You are an expert neuroradiologist analyzing a Brain MRI sequence (FLAIR). "
    "Do NOT interpret this as a CT scan. Ignore Hounsfield Units logic. "
    "Follow these visual rules strictly:\n"
    "1. HYPERINTENSE (Bright/White) regions in the parenchyma indicate EDEMA or TUMOR infiltration, NOT hemorrhage or bone.\n"
    "2. MIXED SIGNAL (High/Low) indicates heterogeneous tumor with NECROSIS.\n"
    "3. Look carefully for MASS EFFECT: Check if the ventricles are compressed or shifted compared to the healthy side.\n"
    "4. Do NOT diagnosis Encephalomalacia unless there is clear volume loss without signal abnormality.\n\n"
    "Task: Describe the lesion location, signal intensity (hyper/hypo), presence of edema, necrosis, and mass effect on ventricles."
)

# Final query prompt (maintains training compatibility)
CANONICAL_PROMPT = "Generate a radiology report for this brain MRI FLAIR scan. \n"

# Lowercase system instruction alias for models that expect `instruction` variable
instruction = INSTRUCTION


# ============================================================================
# NIFTI LOADING / PREPROCESSING (from braingemma3d_architecture.py)
# ============================================================================

def load_nifti_volume(
    nifti_path: str, 
    target_size: Tuple[int, int, int] = (64, 128, 128),
    verbose: bool = True
) -> np.ndarray:
    """
    Loads a NIfTI volume, normalizes, and resizes to target_size (D,H,W).
    Returns numpy array (D,H,W) float32 normalized [0,1].
    
    Following braingemma3d_architecture.py:
    - Canonical orientation (RAS) with nib.as_closest_canonical
    - BraTS handling (H,W,D) → transpose to (D,H,W)
    - HealthyBrains handling (D,H,W) → flip depth axis if upside-down
    - Robust normalization with 1st-99th percentiles
    - Resize to target_size via trilinear interpolation
    
    Args:
        nifti_path: Path to NIfTI file (.nii or .nii.gz)
        target_size: Final dimensions (D, H, W) - default (64, 128, 128) for M3D-LaMed
        verbose: Print debug information
    
    Returns:
        numpy array (D,H,W) float32 normalized [0,1]
    """
    # Resolve path
    p = Path(nifti_path).expanduser()
    alt_cwd = Path.cwd() / nifti_path
    alt_repo = Path(__file__).resolve().parent.parent / nifti_path

    if not p.exists():
        if alt_cwd.exists():
            p = alt_cwd
        elif alt_repo.exists():
            p = alt_repo

    if not p.exists():
        tried = [str(nifti_path), str(alt_cwd), str(alt_repo)]
        raise FileNotFoundError(f"No such file: '{nifti_path}'. Tried: {tried}")

    if verbose:
        print(f"📂 Loading NIfTI: {p}")

    # 1) Load and reorient
    img = nib.load(str(p))
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    
    if verbose:
        print(f"   Original shape: {vol.shape}")
    
    # 2) DETECTION: BraTS vs HealthyBrains
    is_healthy = "HealthyBrains" in nifti_path or "healthy" in nifti_path.lower()
    
    if is_healthy:
        # HealthyBrains: already (D,H,W) but needs depth flip
        if verbose:
            print("   Detected: HealthyBrains (preprocessed) - flipping depth axis")
        vol = np.flip(vol, axis=0).copy()
    else:
        # BraTS: (H,W,D) → (D,H,W)
        if verbose:
            print("   Detected: BraTS - transposing (H,W,D) → (D,H,W)")
        vol = np.transpose(vol, (2, 0, 1))
    
    if verbose:
        print(f"   After orientation fix: {vol.shape} (D,H,W)")
    
    # Robust normalization using percentiles to handle outliers
    vmin, vmax = np.percentile(vol, [1, 99])
    if verbose:
        print(f"   Normalize: percentile [1%, 99%] = [{vmin:.2f}, {vmax:.2f}]")
    
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
        vol = np.clip(vol, 0, 1)
    else:
        if verbose:
            print("   ⚠️ Warning: uniform intensity, setting to 0")
        vol = np.zeros_like(vol)
    
    # 4) Resize to target_size via trilinear interpolation
    # Convert to torch for interpolate, then back to numpy
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    D, H, W = target_size
    vol_t = F.interpolate(vol_t, size=(D, H, W), mode="trilinear", align_corners=False)
    vol_resized = vol_t.squeeze(0).squeeze(0).numpy()  # (D,H,W)
    
    if verbose:
        print(f"   Resized to: {vol_resized.shape}")
        print(f"   Value range: [{vol_resized.min():.3f}, {vol_resized.max():.3f}]")
    
    return vol_resized.astype(np.float32)


# ============================================================================
# REPORT CLEANING (from medgemma_baseline.py)
# ============================================================================

def clean_report(text: str) -> str:
    """
    Remove common headers added by the model (FINDINGS, IMPRESSION, etc.)
    and return only the report body.
    """
    # Pattern per rimuovere header tipo "FINDINGS:", "IMPRESSION:", etc.
    patterns = [
        r'^FINDINGS:\s*',
        r'^IMPRESSION:\s*',
        r'^FINDINGS\s*',
        r'^IMPRESSION\s*',
        r'^CLINICAL HISTORY:\s*',
        r'^TECHNIQUE:\s*',
    ]
    
    cleaned = text.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove multiple empty lines
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    
    return cleaned.strip()


# ============================================================================
# INFERENCE CON M3D-LAMED
# ============================================================================

def generate_report(
    model_name_or_path: str,
    nifti_path: str,
    target_size: Tuple[int, int, int] = (32, 256, 256),
    proj_out_num: int = 256,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
    enable_seg: bool = False,
    question: str = "Can you provide a caption consists of findings for this medical image?",
) -> str:
    """
    Generate a radiology report from a NIfTI file using M3D-LaMed.
    
    Args:
        model_name_or_path: Hugging Face model ID or local path
        nifti_path: Path to NIfTI file
        target_size: Target dimensions (D,H,W) - default (32,256,256) for M3D-LaMed
        proj_out_num: Number of projection tokens (default: 256)
        max_new_tokens: Maximum report length
        do_sample: True for sampling (creative), False for greedy
        temperature: Temperature for sampling
        top_p: Nucleus sampling
        device: PyTorch device (None = auto)
        dtype: Data type (None = bfloat16)
        verbose: Print information
        enable_seg: Enable segmentation
        question: Question for the model
    
    Returns:
        Generated report (string)
    """
    
    # 1) Setup device e dtype
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    if dtype is None:
        dtype = torch.bfloat16
    
    if verbose:
        print("=" * 70)
        print("🔬 M3D-LaMed Baseline Inference")
        print("=" * 70)
        print(f"Model: {model_name_or_path}")
        print(f"Input: {nifti_path}")
        print(f"Target size: {target_size} (D,H,W)")
        print(f"Projection tokens: {proj_out_num}")
        print(f"Max tokens: {max_new_tokens} | Sampling: {do_sample}")
        if do_sample:
            print(f"Temperature: {temperature} | Top-p: {top_p}")
        print(f"Device: {device} | Dtype: {dtype}")
        print(f"Question: {question}")
        print()
    
    # 2) Load model and tokenizer
    if verbose:
        print("📦 Loading model and tokenizer...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    
    model = model.to(device=device)
    
    if verbose:
        print(f"   Model loaded on: {device}")
        print()

    # 3) Load and preprocess NIfTI as .npy
    if verbose:
        print("🧠 Loading and preprocessing NIfTI volume...")
    
    image_np = load_nifti_volume(nifti_path, target_size=target_size, verbose=verbose)
    
    if verbose:
        print(f"   Final array shape: {image_np.shape}")
        print(f"   Value range: [{image_np.min():.3f}, {image_np.max():.3f}]")
        print()
    
    # 4) Prepare input prompt
    if verbose:
        print("🔤 Building prompt with image tokens...")
    
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
    
    # Correct: only unsqueeze(0) for batch dimension
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
    
    if verbose:
        print(f"   Input IDs shape: {input_id.shape}")
        print(f"   Image tensor shape: {image_pt.shape}")
        print()
    
    # 5) Genera report
    if verbose:
        print("🚀 Generating report...")
    
    with torch.inference_mode():
        if enable_seg:
            generation, seg_logit = model.generate(
                image_pt, 
                input_id, 
                seg_enable=True,
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample, 
                top_p=top_p, 
                temperature=temperature
            )
        else:
            generation = model.generate(
                image_pt, 
                input_id, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample, 
                top_p=top_p, 
                temperature=temperature
            )
            seg_logit = None

        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        report = generated_texts[0]
    
    if verbose:
        print("✅ Generation complete!")
        print()
    
    return report


# ============================================================================
# BATCH INFERENCE
# ============================================================================

def batch_inference(
    model_name_or_path: str,
    nifti_paths: List[str],
    output_dir: str,
    **inference_kwargs
) -> List[Tuple[str, str]]:
    """
    Run inference on multiple NIfTI files.
    
    Args:
        model_name_or_path: Hugging Face model ID or local path
        nifti_paths: List of paths to NIfTI files
        output_dir: Directory to save reports
        **inference_kwargs: Parameters for generate_report()
    
    Returns:
        List of tuples (nifti_path, report)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, nifti_path in enumerate(nifti_paths, 1):
        print(f"\n{'=' * 70}")
        print(f"Processing {i}/{len(nifti_paths)}: {Path(nifti_path).name}")
        print(f"{'=' * 70}")
        
        try:
            report = generate_report(model_name_or_path, nifti_path, **inference_kwargs)
            
            # Save report
            output_file = output_dir / f"{Path(nifti_path).stem}_report.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"File: {nifti_path}\n")
                f.write(f"Model: {model_name_or_path}\n")
                f.write(f"{'-' * 70}\n")
                f.write(report)
            
            results.append((nifti_path, report))
            
            print(f"\n📄 Generated report:\n{report}")
            print(f"\n💾 Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            results.append((nifti_path, f"ERROR: {e}"))
    
    # Save summary
    summary_file = output_dir / "inference_summary.csv"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("file_path,report_length,status\n")
        for path, report in results:
            status = "success" if not report.startswith("ERROR") else "error"
            f.write(f"{path},{len(report)},{status}\n")
    
    print(f"\n✅ Batch inference completata! Summary: {summary_file}")
    return results


# ============================================================================
# CLI
# ============================================================================

def build_argparser():
    parser = argparse.ArgumentParser(
        description="M3D-LaMed Baseline - Generate reports from NIfTI files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""  
Examples:
    # Single file
    # Directory (batch)
    python m3d_baseline.py --nifti_dir /path/to/scans/ --output_dir results/
    
    # With custom parameters
    python m3d_baseline.py --nifti_path scan.nii.gz --max_new_tokens 512 --temperature 0.7
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--nifti_path",
        type=str,
        help="Path to single NIfTI file (.nii or .nii.gz)"
    )
    input_group.add_argument(
        "--nifti_dir",
        type=str,
        help="Directory containing NIfTI files (batch inference)"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_output_m3d/",
        help="Output directory for generated reports (default: ./inference_output_m3d/)"
    )
    
    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="GoodBaiBai88/M3D-LaMed-Llama-2-7B",
        help="Hugging Face model ID or local path (default: GoodBaiBai88/M3D-LaMed-Llama-2-7B)"
    )
    
    # Preprocessing
    parser.add_argument(
        "--target_depth",
        type=int,
        default=32,
        help="Target depth (D) for 3D volume (default: 32)"
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=256,
        help="Target height (H) for 3D volume (default: 256)"
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=256,
        help="Target width (W) for 3D volume (default: 256)"
    )
    
    # M3D-LaMed config
    parser.add_argument(
        "--proj_out_num",
        type=int,
        default=256,
        help="Number of projection tokens <im_patch> (default: 256)"
    )
    
    # Generazione
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max length of generated report in tokens (default: 256)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9)"
    )
    
    # Question
    parser.add_argument(
        "--question",
        type=str,
        default="Can you provide a caption consists of findings for this medical image?",
        help="Question for the model (default: caption)"
    )
    
    # Segmentation
    parser.add_argument(
        "--enable_seg",
        action="store_true",
        help="Enable segmentation output"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device PyTorch (default: auto - cuda se disponibile)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type per il modello (default: bfloat16)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed information (default: True)"
    )
    
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Prepare inference parameters
    target_size = (args.target_depth, args.target_height, args.target_width)
    
    inference_kwargs = {
        "target_size": target_size,
        "proj_out_num": args.proj_out_num,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.no_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": args.device,
        "dtype": dtype,
        "verbose": args.verbose,
    }
    
    # Singolo file o batch
    if args.nifti_path:
        # Singolo file
        print("🔬 M3D-LaMed Baseline - Single File Inference")
        print(f"Input: {args.nifti_path}")
        print()
        
        report = generate_report(args.model_name_or_path, args.nifti_path, **inference_kwargs)
        
        # Save output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{Path(args.nifti_path).stem}_generated.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {args.nifti_path}\n")
            f.write(f"Model: {args.model_name_or_path}\n")
            f.write(f"Target size: {target_size}\n")
            f.write(f"{'-' * 70}\n")
            f.write(report)
        
        print("\n" + "=" * 70)
        print("📄 GENERATED REPORT:")
        print("=" * 70)
        print(report)
        print("=" * 70)
        print(f"\n💾 Report saved to: {output_file}")
        
    else:
        # Batch inference
        print("🔬 M3D-LaMed Baseline - Batch Inference")
        print(f"Input directory: {args.nifti_dir}")
        print()
        
        # Find all NIfTI files
        nifti_dir = Path(args.nifti_dir)
        nifti_files = list(nifti_dir.glob("**/*.nii")) + list(nifti_dir.glob("**/*.nii.gz"))
        
        if not nifti_files:
            print(f"❌ Nessun file NIfTI trovato in {args.nifti_dir}")
            sys.exit(1)
        
        print(f"Trovati {len(nifti_files)} file NIfTI")
        
        batch_inference(
            args.model_name_or_path,
            [str(f) for f in nifti_files],
            args.output_dir,
            **inference_kwargs
        )


if __name__ == "__main__":
    main()
