"""
Script to preprocess all brains in the HealthyBrains folder
- Skull stripping with ANTsPy (fast and robust)
- Reorientation and axis adjustment
- Square padding (H == W)
- Robust normalization with percentiles
- Trilinear resize with PyTorch
- Final dimensions: (64, 128, 128)

REQUIREMENTS:
----------
pip install nibabel numpy matplotlib antspyx torch scipy scikit-image

NOTE: 
- ANTsPy (antspyx) is REQUIRED for skull stripping
- PyTorch is RECOMMENDED for high-quality trilinear resize
- Scipy is used as fallback if PyTorch is not available
"""

import os
import glob
import math
import shutil
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # For saving figures without display
import matplotlib.pyplot as plt
from pathlib import Path

# Conditional imports for main dependencies
try:
    import ants
    USE_ANTS = True
    print("✅ ANTsPy loaded correctly")
except ImportError:
    USE_ANTS = False
    print("⚠️ ANTsPy not found. Skull stripping not available.")
    
try:
    import torch
    import torch.nn.functional as F
    USE_TORCH = True
    print("✅ PyTorch loaded correctly")
except ImportError:
    USE_TORCH = False
    print("⚠️ PyTorch not found. Trilinear resize not available.")

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def prepare_nifti_for_ants(nifti_path: str) -> str:
    """
    Prepare NIfTI file for ANTsPy:
    - Create temporary local copy
    - Apply canonical orientation
    """
    nifti_path = os.path.abspath(nifti_path)
    
    # Create temporary directory in the same folder
    tmp_dir = Path(nifti_path).parent / "tmp_ants"
    tmp_dir.mkdir(exist_ok=True)
    
    local_path = str(tmp_dir / f"tmp_{Path(nifti_path).name}")
    
    # Copy and reorient
    shutil.copy(nifti_path, local_path)
    
    img = nib.load(local_path)
    img = nib.as_closest_canonical(img)
    nib.save(img, local_path)
    
    return local_path


def skull_stripping_antspy_fast(nifti_path: str) -> ants.ANTsImage:
    """
    Fast skull stripping with ANTsPy:
    - Downsampling for speed
    - N4 bias correction
    - Brain mask with robust morphology
    - Mask upsampling
    """
    if not USE_ANTS:
        raise ImportError("ANTsPy not available")
    
    print("   ... Skull Stripping ANTsPy (FAST) ...")
    
    ants_path = prepare_nifti_for_ants(nifti_path)
    
    img = ants.image_read(ants_path)
    img = ants.reorient_image2(img, "RAS")
    
    # Downsample for speed (factor 2)
    print("   ... Downsampling per speed-up ...")
    img_ds = ants.resample_image(img, (2, 2, 2), use_voxels=False)
    
    print("   ... N4 Bias Correction ...")
    img_ds_n4 = ants.n4_bias_field_correction(img_ds)
    
    print("   ... Brain Mask Extraction ...")
    mask_ds = ants.get_mask(img_ds_n4, cleanup=2)
    
    # Robust morphology (no holes, better skull removal)
    print("   ... Morphology (closing + opening + fill holes) ...")
    mask_ds = ants.morphology(mask_ds, "close", radius=2)
    mask_ds = ants.morphology(mask_ds, "open", radius=1)
    mask_ds = ants.iMath(mask_ds, "FillHoles")
    mask_ds = ants.morphology(mask_ds, "dilate", radius=1)
    
    # Upsample mask to original resolution
    print("   ... Upsample mask ...")
    mask = ants.resample_image_to_target(
        mask_ds, img, interp_type="nearestNeighbor"
    )
    
    brain = ants.mask_image(img, mask)
    
    # Cleanup temporary files
    try:
        tmp_dir = Path(ants_path).parent
        if tmp_dir.name == "tmp_ants":
            shutil.rmtree(tmp_dir)
    except:
        pass
    
    return brain


def pad_to_square(vol: np.ndarray) -> np.ndarray:
    """
    Symmetric padding to make H == W
    Input : (D, H, W)
    Output: (D, S, S) where S = max(H, W)
    """
    D, H, W = vol.shape
    S = max(H, W)
    
    pad_h = S - H
    pad_w = S - W
    
    pad_h0 = pad_h // 2
    pad_h1 = pad_h - pad_h0
    pad_w0 = pad_w // 2
    pad_w1 = pad_w - pad_w0
    
    vol = np.pad(
        vol,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)),
        mode="constant",
        constant_values=0
    )
    return vol

def save_montage(data, output_path, title="Montage", vmax=None):
    """
    Save a slice montage as PNG image
    With origin="upper" for correct orientation
    """
    n_slices = data.shape[0]
    cols = 8
    rows = math.ceil(n_slices / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.5 * rows), facecolor='black')
    if rows == 1: 
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < n_slices:
            # origin="upper" for correct orientation (slice 0 at bottom)
            axes[i].imshow(data[i], cmap='gray', origin='upper', vmin=0, vmax=vmax)
            axes[i].set_title(f"z={i}", color='cyan', fontsize=9, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, color='white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"   💾 Montage saved: {output_path}")

def process_mri(input_path, output_path, montage_path=None, target_shape=(64, 128, 128)):
    """
    Process a single MRI scan with the following steps:
    1. Skull stripping with ANTsPy
    2. Reorientation to LPS
    3. Axis transposition (D, H, W) 
    4. Slice inversion and visual correction
    5. Square padding (H == W)
    6. Robust normalization with percentiles
    7. Trilinear resize with PyTorch
    8. NIfTI saving
    """
    print(f"\n📂 Processing: {os.path.basename(input_path)}")
    
    try:
        # ===== STEP 1: Skull Stripping con ANTsPy =====
        brain_ants = skull_stripping_antspy_fast(input_path)
        
        # ===== STEP 2: Reorientation to LPS =====
        print("   ... Reorientation to LPS ...")
        brain_ants = ants.reorient_image2(brain_ants, "LPS")
        
        vol = brain_ants.numpy()
        print(f"   Shape after ANTs: {vol.shape}")
        
        # ===== STEP 3: Axis Rearrangement =====
        # In ANTs LPS: 0=Left-Right, 1=Posterior-Anterior, 2=Inferior-Superior
        # Target: 0=Slices (I->S), 1=H (P->A), 2=W (L->R)
        print("   ... Axis transposition ...")
        vol = np.transpose(vol, (2, 1, 0))
        
        # ===== STEP 4: Fix Slice Orientation =====
        # Invert slices to have slice 0 at bottom and last at top
        print("   ... Fix slice orientation ...")
        vol = vol[::-1, :, :]  # Reverse slice order
        vol = np.flip(vol, axis=1)  # Visual correction for vertical axis
        
        print(f"   Shape after transposition: {vol.shape}")
        
        # ===== STEP 5: Square Padding =====
        print("   ... Square padding (H == W) ...")
        vol = pad_to_square(vol)
        print(f"   Shape after padding: {vol.shape}")
        
        # ===== STEP 6: Robust Normalization =====
        print("   ... Robust normalization ...")
        brain_voxels = vol[vol > 0]
        if len(brain_voxels) > 0:
            vmin, vmax = np.percentile(brain_voxels, [1, 99.5])
            vol = np.clip(vol, vmin, vmax)
            vol = (vol - vmin) / (vmax - vmin + 1e-6)
        vol[vol < 0] = 0
        
        # ===== STEP 7: Trilinear Resize with PyTorch =====
        if USE_TORCH:
            print(f"   ... Trilinear resize to {target_shape} ...")
            vol_t = torch.from_numpy(vol.copy())[None, None].float()
            vol_t = F.interpolate(
                vol_t, 
                size=target_shape, 
                mode="trilinear", 
                align_corners=False
            )
            vol_resized = vol_t.squeeze().numpy()
        else:
            # Fallback with scipy (less precise)
            from scipy.ndimage import zoom
            zoom_factors = [t/c for t, c in zip(target_shape, vol.shape)]
            vol_resized = zoom(vol, zoom_factors, order=3)
        
        print(f"   Final shape: {vol_resized.shape}")
        
        # ===== STEP 8: NIfTI Saving =====
        new_img = nib.Nifti1Image(vol_resized, affine=np.eye(4))
        nib.save(new_img, output_path)
        print(f"   ✅ Saved: {output_path}")
        
        # ===== STEP 9: Montage Saving (Optional) =====
        if montage_path:
            save_montage(
                vol_resized, 
                montage_path,
                title=f"{os.path.basename(input_path)} - {target_shape}",
                vmax=1.0  # Normalizzato 0-1
            )
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN FUNCTION: BATCH PROCESSING
# ============================================================================

def load_and_preprocess_single_flair(nifti_path, target_size=(64, 128, 128)):
    """
    Helper function to process a single FLAIR file
    Returns the preprocessed volume as numpy array
    Useful for testing or inference
    """
    brain_ants = skull_stripping_antspy_fast(nifti_path)
    brain_ants = ants.reorient_image2(brain_ants, "LPS")
    
    vol = brain_ants.numpy()
    vol = np.transpose(vol, (2, 1, 0))
    vol = vol[::-1, :, :]
    vol = np.flip(vol, axis=1)
    vol = pad_to_square(vol)
    
    # Normalization
    brain_voxels = vol[vol > 0]
    if len(brain_voxels) > 0:
        vmin, vmax = np.percentile(brain_voxels, [1, 99.5])
        vol = np.clip(vol, vmin, vmax)
        vol = (vol - vmin) / (vmax - vmin + 1e-6)
    vol[vol < 0] = 0
    
    # Resize
    if USE_TORCH:
        vol_t = torch.from_numpy(vol.copy())[None, None].float()
        vol_t = F.interpolate(vol_t, size=target_size, mode="trilinear", align_corners=False)
        return vol_t.squeeze().numpy()
    else:
        from scipy.ndimage import zoom
        zoom_factors = [t/c for t, c in zip(target_size, vol.shape)]
        return zoom(vol, zoom_factors, order=3)


def visualize_flair_grid(nifti_path: str, ncols=8, save_path=None):
    """
    Visualize all slices of a preprocessed FLAIR scan in a grid layout.
    Useful for visually validating preprocessing results.
    """
    vol = load_and_preprocess_single_flair(nifti_path)
    D = vol.shape[0]
    nrows = int(np.ceil(D / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), facecolor="black")
    axes = axes.flatten()
    
    for i in range(D):
        axes[i].imshow(vol[i], cmap="gray", origin="upper")
        axes[i].set_title(f"z={i}", color="cyan", fontsize=9, fontweight='bold')
        axes[i].axis("off")
    
    for i in range(D, len(axes)): 
        axes[i].axis("off")
    
    plt.suptitle(f"Preprocessed: {os.path.basename(nifti_path)}", 
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='black')
        print(f"💾 Visualization saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def process_all_healthy_brains(
    input_dir="/leonardo_work/CESMA_leonardo/CBMS/Datasets/HealthyBrains",
    output_dir="/leonardo_work/CESMA_leonardo/CBMS/Datasets/HealthyBrains_Preprocessed",
    target_shape=(64, 128, 128),
    save_montages=True
):
    """
    Process all brains in the HealthyBrains folder
    
    Args:
        input_dir: Folder with original data
        output_dir: Folder to save results
        target_shape: Final dimensions (depth, height, width)
        save_montages: If True, also save preview images
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_montages:
        montage_dir = output_dir / "montages"
        montage_dir.mkdir(exist_ok=True)
    
    # Find all sub-XXXXXX folders
    subject_dirs = sorted(glob.glob(os.path.join(input_dir, "sub-*")))
    
    print("=" * 80)
    print(f"🧠 PREPROCESSING HEALTHY BRAINS")
    print("=" * 80)
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target shape: {target_shape}")
    print(f"Subjects found: {len(subject_dirs)}")
    print("=" * 80)
    
    # Statistiche
    success_count = 0
    failed_count = 0
    failed_subjects = []
    
    # Process each subject
    for idx, subject_dir in enumerate(subject_dirs, 1):
        subject_name = os.path.basename(subject_dir)
        print(f"\n[{idx}/{len(subject_dirs)}] 🧠 {subject_name}")
        
        try:
            # Find FLAIR file in folder
            flair_files = glob.glob(os.path.join(subject_dir, "*_FLAIR.nii.gz"))
            
            if not flair_files:
                print(f"   ⚠️ No FLAIR file found in {subject_name}")
                failed_count += 1
                failed_subjects.append((subject_name, "No FLAIR file"))
                continue
            
            input_path = flair_files[0]
            
            # Prepara path di output
            output_filename = f"{subject_name}_preprocessed.nii.gz"
            output_path = output_dir / output_filename
            
            montage_path = None
            if save_montages:
                montage_filename = f"{subject_name}_preview.png"
                montage_path = montage_dir / montage_filename
            
            # Processa
            success = process_mri(input_path, output_path, montage_path, target_shape)
            
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_subjects.append((subject_name, "Processing failed"))
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            failed_count += 1
            failed_subjects.append((subject_name, str(e)))
            continue
    
    # Final report
    print("\n" + "=" * 80)
    print("📊 FINAL REPORT")
    print("=" * 80)
    print(f"✅ Successi: {success_count}/{len(subject_dirs)}")
    print(f"❌ Falliti:  {failed_count}/{len(subject_dirs)}")
    
    if failed_subjects:
        print("\n❌ Soggetti falliti:")
        for subj, reason in failed_subjects:
            print(f"   - {subj}: {reason}")
    
    # Save log
    log_path = output_dir / "preprocessing_log.txt"
    with open(log_path, 'w') as f:
        f.write("PREPROCESSING LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input dir: {input_dir}\n")
        f.write(f"Output dir: {output_dir}\n")
        f.write(f"Target shape: {target_shape}\n")
        f.write(f"Total subjects: {len(subject_dirs)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {failed_count}\n\n")
        
        if failed_subjects:
            f.write("Failed subjects:\n")
            for subj, reason in failed_subjects:
                f.write(f"  - {subj}: {reason}\n")
    
    print(f"\n📝 Log saved to: {log_path}")
    print("=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/leonardo_work/CESMA_leonardo/CBMS/Datasets/HealthyBrains"
    OUTPUT_DIR = "/leonardo_work/CESMA_leonardo/CBMS/Datasets/HealthyBrains_Preprocessed"
    TARGET_SHAPE = (64, 128, 128)
    SAVE_MONTAGES = True  # Metti False per velocizzare se non servono preview
    
    # Check dependencies
    if not USE_ANTS:
        print("❌ ERROR: ANTsPy is required for preprocessing!")
        print("   Installalo con: pip install antspyx")
        exit(1)
    
    if not USE_TORCH:
        print("⚠️ WARNING: PyTorch not found, will use scipy (less accurate)")
        print("   Consigliato installare: pip install torch")
    
    # Start processing
    process_all_healthy_brains(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_shape=TARGET_SHAPE,
        save_montages=SAVE_MONTAGES
    )
