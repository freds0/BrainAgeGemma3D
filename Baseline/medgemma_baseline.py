#!/usr/bin/env python3
"""
MedGemma Baseline Script - Approach with NIfTI support

This script adapts the MedGemma (PaliGemma) model to generate radiological reports
from NIfTI files (3D MRI volumes), extracting representative 2D slices.

Features:
- NIfTI file loading (.nii, .nii.gz)
- Canonical orientation (RAS) with BraTS/HealthyBrains handling
- Equidistant slice extraction along the axial axis
- Robust normalization (1st-99th percentiles)
- Conversion to PIL RGB for PaliGemma
- Canonical prompt: "Generate a radiology report for this brain MRI FLAIR scan. \n"

Uso:
    python medgemma_baseline.py --nifti_path path/to/scan.nii.gz --num_slices 16 --max_new_tokens 256
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
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import base64
import io

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# System instruction for neuroradiology MRI analysis
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

# Query prompt for report generation
CANONICAL_PROMPT = "Generate a radiology report for this brain MRI FLAIR scan. \n"


# ============================================================================
# PREPROCESSING NIFTI → PIL IMAGES
# ============================================================================

def clean_report(text: str) -> str:
    """
    Remove common headers added by the model (FINDINGS, IMPRESSION, etc.)
    and return only the report body.
    """
    import re
    
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


def encode_image_to_base64(img: Image.Image, format: str = "jpeg") -> str:
    """
    Encode PIL image to base64 data URI (official MedGemma format).
    
    Args:
        img: PIL Image to encode
        format: Image format ("jpeg" or "png")
    
    Returns:
        Data URI string in format "data:image/jpeg;base64,..."
    """
    with io.BytesIO() as img_bytes:
        img.save(img_bytes, format=format.upper())
        img_bytes.seek(0)
        encoded_string = base64.b64encode(img_bytes.getbuffer()).decode("utf-8")
    return f"data:image/{format};base64,{encoded_string}"


def load_and_process_nifti(
    file_path: str,
    num_slices: int = 64,
    target_size: Tuple[int, int] = (128, 128),
    normalize_method: str = "percentile",
    verbose: bool = True
) -> List[Image.Image]:
    """
    Load a NIfTI file, normalize, extract equidistant slices, and convert to PIL RGB.
    
    Processing pipeline:
    - Canonical orientation (RAS) using nib.as_closest_canonical
    - BraTS data: (H,W,D) → transpose to (D,H,W)
    - HealthyBrains data: (D,H,W) → flip depth axis if needed
    - Robust normalization using 1st-99th percentiles
    
    Args:
        file_path: Path to NIfTI file (.nii or .nii.gz)
        num_slices: Number of slices to extract (equidistant)
        target_size: Final size (H, W) for each slice
        normalize_method: 'percentile' (default) or 'minmax'
        verbose: Print debug information
    
    Returns:
        List of PIL.Image.Image in RGB format
    """
    # Resolve path: accept absolute or workspace-relative inputs and provide
    # a helpful error message if the file cannot be found.
    p = Path(file_path).expanduser()
    alt_cwd = Path.cwd() / file_path
    alt_repo = Path(__file__).resolve().parent.parent / file_path

    if not p.exists():
        if alt_cwd.exists():
            p = alt_cwd
        elif alt_repo.exists():
            p = alt_repo

    if not p.exists():
        tried = [str(file_path), str(alt_cwd), str(alt_repo)]
        raise FileNotFoundError(f"No such file or no access: '{file_path}'. Tried: {tried}")

    if verbose:
        print(f"📂 Loading NIfTI: {p}")

    # Load NIfTI file
    nifti_img = nib.load(str(p))
    
    # Reorient to canonical orientation (RAS) for consistency
    nifti_img = nib.as_closest_canonical(nifti_img)
    data = nifti_img.get_fdata(dtype=np.float32)
    
    if verbose:
        print(f"   Original shape: {data.shape}")
        print(f"   Original affine:\n{nifti_img.affine}")
    
    # Detect data source: BraTS or HealthyBrains
    is_healthy = "HealthyBrains" in file_path or "healthy" in file_path.lower()
    
    if is_healthy:
        # HealthyBrains preprocessed: already (D,H,W) but upside down → flip
        if verbose:
            print("   Detected: HealthyBrains (preprocessed) - flipping depth axis")
        data = data[::-1, :, :]  # Flip along depth (axis 0)
    else:
        # BraTS: saved as (H,W,D) after as_closest_canonical → transpose
        if verbose:
            print("   Detected: BraTS or standard MRI - transposing (H,W,D) → (D,H,W)")
        data = data.transpose(2, 0, 1)  # (H,W,D) → (D,H,W)
    
    if verbose:
        print(f"   After orientation fix: {data.shape} (D,H,W)")
    
    # Robust normalization using percentiles to handle outliers
    if normalize_method == "percentile":
        vmin, vmax = np.percentile(data, [1, 99])
        if verbose:
            print(f"   Normalize: percentile [1%, 99%] = [{vmin:.2f}, {vmax:.2f}]")
    else:  # minmax
        vmin, vmax = data.min(), data.max()
        if verbose:
            print(f"   Normalize: min-max = [{vmin:.2f}, {vmax:.2f}]")
    
    if vmax > vmin:
        data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1) * 255
    else:
        if verbose:
            print("   ⚠️ Warning: uniform intensity, setting to 128 (gray)")
        data_norm = np.full_like(data, 128)
    
    data_norm = data_norm.astype(np.uint8)
    
    # Select equidistant slices along axial axis (depth)
    D, H, W = data_norm.shape
    total_slices = D
    
    # Generate exactly num_slices equidistant indices
    # If total_slices < num_slices, np.linspace will repeat indices as needed
    if num_slices > total_slices and verbose:
        print(f"   ⚠️ Warning: requested num_slices ({num_slices}) > available ({total_slices}) - indices will repeat")
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    if verbose:
        print(f"   Extracting {len(indices)} slices from {total_slices} available")
        print(f"   Slice indices: {indices}")
    
    # Extract slices and convert to PIL RGB format
    pil_images = []
    for i in indices:
        # Extract 2D slice along axial axis (D)
        slice_2d = data_norm[i, :, :]  # (H, W)
        
        # IMPORTANT: Rotate 90° if needed for correct orientation
        # (depends on MRI acquisition protocol)
        # Add np.rot90(slice_2d, k=1) if images appear rotated
        # For now, keep standard orientation
        
        # Converti in PIL
        img = Image.fromarray(slice_2d, mode='L')  # Grayscale
        
        # Converti in RGB (PaliGemma richiede 3 canali)
        img = img.convert("RGB")
        
        # Resize se richiesto
        if target_size is not None:
            img = img.resize(target_size, resample=Image.BICUBIC)
        
        pil_images.append(img)
    
    if verbose:
        print(f"✅ Processed {len(pil_images)} slices → {target_size} RGB")
    
    return pil_images


# ============================================================================
# INFERENCE CON MEDGEMMA (PaliGemma)
# ============================================================================

def generate_report(
    model_id: str,
    nifti_path: str,
    num_slices: int = 64,
    target_size: Tuple[int, int] = (128, 128),
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Generate a radiology report from a NIfTI file using MedGemma.
    
    Args:
        model_id: Hugging Face model ID o path locale (es. "google/medgemma-2b-v1.1")
        nifti_path: Path al file NIfTI
        num_slices: Numero di slice da estrarre
        target_size: Dimensione slice (H, W)
        max_new_tokens: Lunghezza massima report
        do_sample: True for sampling (creative), False for greedy (deterministic)
        temperature: Temperature for sampling (only if do_sample=True)
        top_p: Nucleus sampling (only if do_sample=True)
        device: PyTorch device (None = auto)
        verbose: Print information
    
    Returns:
        Generated report text
    """
    
    # Load model and processor
    if verbose:
        print("=" * 70)
        print("🔬 MedGemma Baseline Inference")
        print("=" * 70)
        print(f"Model: {model_id}")
        print(f"Input: {nifti_path}")
        print(f"Slices: {num_slices} | Size: {target_size}")
        print(f"Tokens: {max_new_tokens} | Sampling: {do_sample}")
        if do_sample:
            print(f"Temperature: {temperature} | Top-p: {top_p}")
        print()
    
    if verbose:
        print("📦 Loading model and processor...")
    
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # bfloat16 as in official notebook
        device_map="auto" if device is None else device,
        offload_buffers=True,
    )
    
    if verbose:
        print(f"   Model loaded on: {model.device}")
        print()
    
    # 2) Load and preprocess NIfTI
    input_images = load_and_process_nifti(
        nifti_path,
        num_slices=num_slices,
        target_size=target_size,
        verbose=verbose
    )
    
    if verbose:
        print()
        print("🔤 Building chat message with base64 encoded images...")
    
    # 3) Build message in chat format (official MedGemma approach)
    # Pattern: INSTRUCTION → IMAGES (with SLICE labels) → QUERY
    content = []
    
    # System instruction (guides the model for neuroradiology MRI)
    content.append({"type": "text", "text": INSTRUCTION})
    
    # Alternate images with slice labels
    for slice_number, img in enumerate(input_images, 1):
        # Encode image to base64
        img_base64 = encode_image_to_base64(img, format="jpeg")
        content.append({"type": "image", "image": img_base64})
        content.append({"type": "text", "text": f"SLICE {slice_number}"})
    
    # Final query
    content.append({"type": "text", "text": CANONICAL_PROMPT.strip()})
    
    messages = [{"role": "user", "content": content}]
    
    if verbose:
        print(f"   Created message with {len(input_images)} images")
        print(f"   Total content items: {len(content)}")
    
    # Tokenize using apply_chat_template (official method)
    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    
    input_len = model_inputs["input_ids"].shape[-1]
    
    if verbose:
        print(f"   Input IDs shape: {model_inputs['input_ids'].shape}")
        if 'pixel_values' in model_inputs:
            print(f"   Pixel values shape: {model_inputs['pixel_values'].shape}")
        print()
    
    # 4) Genera report
    if verbose:
        print("🚀 Generating report...")
    
    with torch.inference_mode():
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        
        # Generate report
        generation = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
        )
        
        # Post-process output
        medgemma_response = processor.post_process_image_text_to_text(
            generation, skip_special_tokens=True
        )[0]
        
        # Remove input text from response if present
        decoded_inputs = processor.post_process_image_text_to_text(
            model_inputs["input_ids"], skip_special_tokens=True
        )[0]
        
        index_input_text = medgemma_response.find(decoded_inputs)
        if 0 <= index_input_text <= 2:
            medgemma_response = medgemma_response[index_input_text + len(decoded_inputs):]
        
        # Clean report by removing FINDINGS/IMPRESSION headers
        decoded = clean_report(medgemma_response.strip())
    
    if verbose:
        print("✅ Generation complete!")
        print()
    
    return decoded


# ============================================================================
# BATCH INFERENCE (multipli file NIfTI)
# ============================================================================

def batch_inference(
    model_id: str,
    nifti_paths: List[str],
    output_dir: str,
    **inference_kwargs
) -> List[Tuple[str, str]]:
    """
    Inference execution on multiple NIfTI files.
    
    Args:
        model_id: Hugging Face model ID or local path
        nifti_paths: Lists of path to NIfTI files
        output_dir: Directory where to save the reports
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
            report = generate_report(model_id, nifti_path, **inference_kwargs)
            
            # Save report
            output_file = output_dir / f"{Path(nifti_path).stem}_report.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"File: {nifti_path}\n")
                f.write(f"Model: {model_id}\n")
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
        description="MedGemma Baseline - Generate report from NIfTI files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples of usage:
    # Single file
    python medgemma_baseline.py --nifti_path scan.nii.gz --output_dir results/
    
    # Directory (batch)
    python medgemma_baseline.py --nifti_dir /path/to/scans/ --output_dir results/
    
    # With custom parameters
    python medgemma_baseline.py --nifti_path scan.nii.gz --num_slices 32 --max_new_tokens 512 --do_sample
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
        default="./inference_output_baseline/",
        help="Output directory for generated reports (default: ./inference_output_baseline/)"
    )
    
    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="/leonardo_work/CESMA_leonardo/CBMS/Models/medgemma",
        help="Hugging Face model ID or local path (default: local Models/medgemma)"
    )
    
    # Preprocessing
    parser.add_argument(
        "--num_slices",
        type=int,
        default=64,
        help="Number of slices to extract from 3D volume (default: 64)"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Target size (H W) for each slice (default: 128 128)"
    )
    
    # Generazione
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Lunghezza massima report in token (default: 256)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Usa sampling invece di greedy decoding"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0, only if --do_sample)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p (default: 1.0, only if --do_sample)"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device PyTorch (default: auto)"
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
    
    # Prepare inference parameters
    inference_kwargs = {
        "num_slices": args.num_slices,
        "target_size": tuple(args.target_size),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": args.device,
        "verbose": args.verbose,
    }
    
    # Singolo file o batch
    if args.nifti_path:
        # Singolo file
        print("🔬 MedGemma Baseline - Single File Inference")
        print(f"Input: {args.nifti_path}")
        print()
        
        report = generate_report(args.model_id, args.nifti_path, **inference_kwargs)
        
        # Save output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{Path(args.nifti_path).stem}_generated.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {args.nifti_path}\n")
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Prompt: {CANONICAL_PROMPT.strip()}\n")
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
        print("🔬 MedGemma Baseline - Batch Inference")
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
            args.model_id,
            [str(f) for f in nifti_files],
            args.output_dir,
            **inference_kwargs
        )


if __name__ == "__main__":
    main()
