#!/usr/bin/env python3
"""
MD3VLM Baseline - Report Generation for BraTS MRI
==================================================
Baseline per generazione report usando Med3DVLM (MD3VLM).
Uses MD3VLM's native input size (128×256×256).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

# Import preprocessing functions for percentile-based normalization
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from braingemma3d_architecture import load_nifti_volume
    PREPROCESSING_AVAILABLE = True
    print("[OK] Using braingemma3d_architecture preprocessing (percentile normalization)")
except ImportError:
    PREPROCESSING_AVAILABLE = False
    print("[WARN] braingemma3d_architecture not found - using fallback preprocessing")

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# MD3VLM-specific configuration
MD3VLM_IMAGE_SIZE = (128, 256, 256)  # (D, H, W) - native MD3VLM input size
MD3VLM_MAX_LENGTH = 1024
MD3VLM_PROJ_OUT_NUM = 256  # Default projection tokens

# Use MD3VLM native size (not training size)
DEFAULT_TARGET_SIZE = (128, 256, 256)  # MD3VLM native size

DEFAULT_QUESTION = "Generate a radiology report for this brain MRI FLAIR scan. \n"


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def load_and_preprocess_nifti(
    nifti_path: str,
    target_size: Tuple[int, int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    Load and preprocess NIfTI volume with percentile normalization.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        target_size: Target size (D, H, W)
        
    Returns:
        numpy array of shape (D, H, W) ready for model input
    """
    if PREPROCESSING_AVAILABLE:
        # Use percentile normalization from braingemma3d_architecture
        volume = load_nifti_volume(nifti_path, target_size=target_size)
        # load_nifti_volume returns (1, 1, D, H, W), remove batch dims
        if isinstance(volume, torch.Tensor):
            volume = volume.squeeze(0).squeeze(0).cpu().numpy()
        else:
            volume = volume.squeeze(0).squeeze(0)
        return volume
    else:
        # Fallback preprocessing (not recommended for comparison!)
        print("[WARN] Using fallback preprocessing - results may not be comparable!")
        import nibabel as nib
        from scipy.ndimage import zoom
        
        nii_img = nib.load(nifti_path)
        volume = nii_img.get_fdata().astype(np.float32)
        
        # BraTS format (H, W, D) -> (D, H, W)
        if volume.ndim == 3 and volume.shape[2] < volume.shape[0]:
            volume = np.transpose(volume, (2, 0, 1))
        
        # Percentile normalization
        p_low, p_high = np.percentile(volume, [1.0, 99.0])
        volume = np.clip(volume, p_low, p_high)
        if p_high > p_low:
            volume = (volume - p_low) / (p_high - p_low)
        
        # Resize
        zoom_factors = [
            target_size[0] / volume.shape[0],
            target_size[1] / volume.shape[1],
            target_size[2] / volume.shape[2]
        ]
        volume = zoom(volume, zoom_factors, order=1)
        
        return volume


def save_volume_as_npy(volume: np.ndarray, output_path: str):
    """Save preprocessed volume as .npy file for MD3VLM input"""
    # MD3VLM expects shape (1, D, H, W) or (D, H, W)
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]  # Add batch dimension
    
    np.save(output_path, volume)
    print(f"[SAVE] Saved preprocessed volume: {output_path}")


# ============================================================
# MODEL LOADING
# ============================================================

def load_md3vlm_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoProcessor, int]:
    """
    Load MD3VLM model and processor.
    
    Returns:
        model, processor, proj_out_num
    """
    print(f"[LOAD] Loading MD3VLM model from: {model_path}")
    
    # Load processor (preferred) but fall back to tokenizer if processor loading fails
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            max_length=MD3VLM_MAX_LENGTH,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[WARN] AutoProcessor.from_pretrained failed: {e}")
        print("   Falling back to AutoTokenizer (text-only processor).")
        processor = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
        trust_remote_code=True,
    )
    
    # Get projection output number
    proj_out_num = (
        model.get_model().config.proj_out_num
        if hasattr(model.get_model().config, "proj_out_num")
        else MD3VLM_PROJ_OUT_NUM
    )
    
    print(f"[OK] Model loaded on: {DEVICE}")
    print(f"   Projection tokens: {proj_out_num}")
    
    return model, processor, proj_out_num


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    model_path: str,
    nifti_path: str,
    question: str = DEFAULT_QUESTION,
    target_size: Tuple[int, int, int] = DEFAULT_TARGET_SIZE,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    output_npy: Optional[str] = None,
) -> str:
    """
    Generate a report for a single NIfTI file using MD3VLM.
    
    Args:
        model_path: Path to MD3VLM model
        nifti_path: Path to input .nii file
        question: Question/prompt for the model
        target_size: Target volume size (D, H, W) - DEFAULT: (128, 256, 256) MD3VLM native size
        max_new_tokens: Maximum tokens to generate
        do_sample: Use sampling for generation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        output_npy: Optional path to save preprocessed volume
        
    Returns:
        Generated report text
    """
    
    # Load model
    model, processor, proj_out_num = load_md3vlm_model(model_path)
    
    # Preprocess volume
    print(f"[PREPROCESS] Loading and preprocessing NIfTI volume...")
    volume = load_and_preprocess_nifti(nifti_path, target_size=target_size)
    
    print(f"   Final array shape: {volume.shape}")
    print(f"   Value range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Save as .npy if requested
    if output_npy:
        save_volume_as_npy(volume, output_npy)
    
    # Add batch and channel dimensions if needed
    # MD3VLM expects shape: (B, C, D, H, W)
    if volume.ndim == 3:
        volume = volume[np.newaxis, np.newaxis, ...]  # (1, 1, D, H, W)
    elif volume.ndim == 4:
        volume = volume[:, np.newaxis, ...]  # (B, 1, D, H, W)
    
    # Create input prompt with image tokens
    print(f"[PROMPT] Building prompt with image tokens...")
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    
    input_ids = processor(input_txt, return_tensors="pt")["input_ids"].to(device=DEVICE)
    print(f"   Input IDs shape: {input_ids.shape}")
    
    # Convert numpy to tensor
    image_tensor = torch.from_numpy(volume).to(dtype=DTYPE, device=DEVICE)
    print(f"   Image tensor shape: {image_tensor.shape}")
    
    # Generate report
    print(f"[GENERATE] Generating report...")
    with torch.no_grad():
        generation = model.generate(
            images=image_tensor,
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )
    
    # Decode output
    report = processor.decode(generation[0], skip_special_tokens=True)
    
    print(f"[OK] Generation complete!")
    
    return report


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_generate_reports(
    model_path: str,
    brats_base_dir: str,
    output_dir: str,
    modality: str = "flair",
    question: str = DEFAULT_QUESTION,
    max_patients: Optional[int] = None,
    target_size: Tuple[int, int, int] = DEFAULT_TARGET_SIZE,
    save_npy: bool = False,
    **generation_kwargs
) -> Dict[str, str]:
    """
    Batch process multiple BraTS patients and generate reports.
    
    Args:
        model_path: Path to MD3VLM model
        brats_base_dir: Base directory containing BraTS patient folders
        output_dir: Directory to save generated reports
        modality: MRI modality (flair, t1, t1ce, t2)
        question: Question prompt
        max_patients: Maximum number of patients to process
        target_size: Target volume size (D, H, W)
        save_npy: Whether to save preprocessed .npy files
        **generation_kwargs: Additional arguments for generation
        
    Returns:
        Dictionary mapping patient_id -> generated report
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all patient directories
    brats_path = Path(brats_base_dir)
    patient_dirs = sorted([d for d in brats_path.iterdir() if d.is_dir()])
    
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
    
    print(f"\n[BATCH] Batch processing {len(patient_dirs)} patients...")
    
    # Load model once
    model, processor, proj_out_num = load_md3vlm_model(model_path)
    
    results = {}
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        
        # Find NIfTI file
        nifti_path = patient_dir / f"{patient_id}_{modality}.nii"
        if not nifti_path.exists():
            print(f"[WARN] Skipping {patient_id}: {modality} file not found")
            continue
        
        try:
            # Preprocess volume
            volume = load_and_preprocess_nifti(str(nifti_path), target_size=target_size)
            
            # Save .npy if requested
            if save_npy:
                npy_path = os.path.join(output_dir, f"{patient_id}_{modality}.npy")
                save_volume_as_npy(volume, npy_path)
            
            # Add batch and channel dimensions (B, C, D, H, W)
            if volume.ndim == 3:
                volume = volume[np.newaxis, np.newaxis, ...]
            elif volume.ndim == 4:
                volume = volume[:, np.newaxis, ...]
            
            # Create input
            image_tokens = "<im_patch>" * proj_out_num
            input_txt = image_tokens + question
            input_ids = processor(input_txt, return_tensors="pt")["input_ids"].to(device=DEVICE)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(volume).to(dtype=DTYPE, device=DEVICE)
            
            # Generate
            with torch.no_grad():
                generation = model.generate(
                    images=image_tensor,
                    inputs=input_ids,
                    **generation_kwargs
                )
            
            report = processor.decode(generation[0], skip_special_tokens=True)
            
            # Save report
            report_path = os.path.join(output_dir, f"{patient_id}_{modality}_generated.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            
            results[patient_id] = report
            
        except Exception as e:
            print(f"[ERROR] Error processing {patient_id}: {e}")
            results[patient_id] = f"ERROR: {str(e)}"
    
    # Save summary
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Batch processing complete!")
    print(f"   Processed: {len(results)}/{len(patient_dirs)} patients")
    print(f"   Results saved to: {output_dir}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MD3VLM Baseline - Report Generation")
    
    # Model and input
    parser.add_argument("--model-path", type=str, 
                       default="/leonardo_work/CESMA_leonardo/CBMS/Models/md3vlm",
                       help="Path to MD3VLM model")
    parser.add_argument("--nifti-path", type=str, default=None,
                       help="Path to single NIfTI file (for single file inference)")
    parser.add_argument("--brats-dir", type=str, default=None,
                       help="Path to BraTS dataset directory (for batch processing)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="inference_output_md3vlm",
                       help="Output directory for generated reports")
    parser.add_argument("--save-npy", action="store_true",
                       help="Save preprocessed volumes as .npy files")
    
    # Processing
    parser.add_argument("--modality", type=str, default="flair",
                       choices=["flair", "t1", "t1ce", "t2"],
                       help="MRI modality")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION,
                       help="Question/prompt for the model")
    parser.add_argument("--max-patients", type=int, default=None,
                       help="Maximum number of patients to process (for batch)")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling top-p")
    parser.add_argument("--no-sampling", action="store_true",
                       help="Use greedy decoding instead of sampling")
    
    # Image preprocessing
    parser.add_argument("--target-size", type=int, nargs=3, default=[128, 256, 256],
                       help="Target volume size (D H W) - DEFAULT: (128, 256, 256) MD3VLM native size")
    
    args = parser.parse_args()
    
    target_size = tuple(args.target_size)
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.no_sampling,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    # Single file inference
    if args.nifti_path:
        print("\n" + "="*70)
        print("[MD3VLM] Baseline - Single File Inference")
        print("="*70)
        print(f"[INFO] Using MD3VLM native size: {target_size}")
        print(f"Input: {args.nifti_path}")
        
        output_npy = None
        if args.save_npy:
            patient_id = Path(args.nifti_path).stem.replace('.nii', '')
            output_npy = os.path.join(args.output_dir, f"{patient_id}.npy")
        
        report = generate_report(
            model_path=args.model_path,
            nifti_path=args.nifti_path,
            question=args.question,
            target_size=target_size,
            output_npy=output_npy,
            **generation_kwargs
        )
        
        print("\n" + "="*70)
        print("[REPORT] GENERATED REPORT:")
        print("="*70)
        print(report)
        print("="*70)
        
        # Save report
        os.makedirs(args.output_dir, exist_ok=True)
        patient_id = Path(args.nifti_path).stem.replace('.nii', '')
        report_path = os.path.join(args.output_dir, f"{patient_id}_generated.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"[SAVE] Report saved to: {report_path}")
    
    # Batch processing
    elif args.brats_dir:
        print("\n" + "="*70)
        print("[MD3VLM] Baseline - Batch Processing")
        print("="*70)
        print(f"[INFO] Using MD3VLM native size: {target_size}")
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.brats_dir}")
        print(f"Modality: {args.modality}")
        print(f"Target size: {target_size} (D,H,W)")
        print(f"Max tokens: {args.max_new_tokens} | Sampling: {not args.no_sampling}")
        print(f"Temperature: {args.temperature} | Top-p: {args.top_p}")
        print(f"Device: {DEVICE} | Dtype: {DTYPE}")
        
        batch_generate_reports(
            model_path=args.model_path,
            brats_base_dir=args.brats_dir,
            output_dir=args.output_dir,
            modality=args.modality,
            question=args.question,
            max_patients=args.max_patients,
            target_size=target_size,
            save_npy=args.save_npy,
            **generation_kwargs
        )
    
    else:
        print("[ERROR] Error: Must specify either --nifti-path or --brats-dir")
        parser.print_help()


if __name__ == "__main__":
    main()
