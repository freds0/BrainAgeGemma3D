#!/usr/bin/env python3
"""
BrainGemma3D Architecture
=========================
Definition of the BrainGemma3D model architecture:
- 3D Vision Encoder (inflated from SigLIP 2D)
- Language Model (quantized MedGemma)
- Vision-Language Projection
- Utilities for loading models and NIfTI volumes
"""

# ENVIRONMENT CONFIGURATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================
# PROMPT STRUCTURE - CANONICAL (NO AUGMENTATION)
# ============================================================
CANONICAL_PROMPT = "Generate a radiology report for this brain MRI FLAIR scan. \n"


def get_training_prompt(patient_id: str = None) -> str:
    """Returns CANONICAL_PROMPT (backward compatibility)"""
    return CANONICAL_PROMPT


def get_inference_prompt(user_prompt: str = None) -> str:
    """Returns CANONICAL_PROMPT (backward compatibility)"""
    return CANONICAL_PROMPT


# ============================================================
# MODEL LOADERS (Local)
# ============================================================

def load_sigclip_local(sigclip_dir: str):
    """
    Load SigLIP/MedSigLIP from local directory.
    
    Args:
        sigclip_dir: Path to SigLIP model directory
        
    Returns:
        Loaded SigLIP model
    """
    if not os.path.isdir(sigclip_dir):
        raise FileNotFoundError(f"SigLIP directory not found: {sigclip_dir}")
    base = AutoModel.from_pretrained(sigclip_dir, local_files_only=True)
    return base


def load_medgemma_lm_local(medgemma_dir: str, device_map=None):
    """
    Load MedGemma CausalLM with 4-bit quantization.
    
    Args:
        medgemma_dir: Path to MedGemma model directory
        device_map: Device mapping for model placement
        
    Returns:
        Quantized MedGemma language model
    """
    if not os.path.isdir(medgemma_dir):
        raise FileNotFoundError(f"MedGemma directory not found: {medgemma_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else None

    lm = AutoModelForCausalLM.from_pretrained(
        medgemma_dir,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return lm


def load_medgemma_tokenizer_local(medgemma_dir: str):
    """
    Load MedGemma tokenizer from local directory.
    
    Args:
        medgemma_dir: Path to MedGemma model directory
        
    Returns:
        MedGemma tokenizer
    """
    if not os.path.isdir(medgemma_dir):
        raise FileNotFoundError(f"MedGemma directory not found: {medgemma_dir}")
    tok = AutoTokenizer.from_pretrained(medgemma_dir, local_files_only=True)
    return tok


# ============================================================
# NIFTI LOADING / PREPROCESSING
# ============================================================

def load_nifti_volume(nifti_path: str, target_size: Tuple[int, int, int] = (64, 128, 128)) -> torch.Tensor:
    """
    Load a NIfTI volume, normalize, and resize to target_size (D,H,W).
    Returns tensor (1,1,D,H,W) float32 on CPU.

    Note: 
    - BraTS NIfTI files are saved as (H,W,D) after as_closest_canonical, so we transpose to (D,H,W)
    - HealthyBrains preprocessed NIfTI are already (D,H,W) but UPSIDE DOWN -> flip depth axis
    
    Args:
        nifti_path: Path to NIfTI file
        target_size: Target dimensions (depth, height, width)
        
    Returns:
        Volume tensor (1,1,D,H,W)
    """
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    
    # DETECTION: HealthyBrains preprocessed vs BraTS original
    is_healthy = "HealthyBrains" in nifti_path or "healthy" in nifti_path.lower()
    
    if is_healthy:
        # Healthy: already (D,H,W) but flip height axis
        vol = np.flip(vol, axis=1).copy()
    else:
        # BraTS: (H,W,D) -> (D,H,W)
        vol = np.transpose(vol, (2, 1, 0))

    # Robust normalization
    vmin, vmax = np.percentile(vol, 1), np.percentile(vol, 99)
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
        vol = np.clip(vol, 0, 1)
    else:
        vol = np.zeros_like(vol)

    # Convert to torch: (D,H,W)
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    # Resize via trilinear interpolation
    D, H, W = target_size
    vol_t = F.interpolate(vol_t, size=(D, H, W), mode="trilinear", align_corners=False)

    return vol_t.contiguous().float()


def load_nifti_volume_general(
    nifti_path: str,
    target_size: Tuple[int, int, int] = (64, 128, 128),
    verbose: bool = True,
) -> torch.Tensor:
    """
    General-purpose NIfTI loader for brain MRI volumes from any dataset.

    Automatically detects the depth (axial/slice) axis without relying on
    file-path naming conventions. Works for:
      - Standard datasets whose NIfTI headers correctly encode orientation
        (e.g. BraTS, any BIDS-compliant dataset).
      - Datasets whose volumes were saved with an identity affine (e.g. the
        HealthyBrains dataset in this project) where the depth axis is
        implicitly the smallest spatial dimension.

    Detection strategy (in order of priority):
      1. **Voxel-size heuristic** (most reliable for multi-modal MRI):
         After ``as_closest_canonical``, read the three voxel sizes (zooms).
         The axis whose voxel pitch is >= 1.5x the median of the other two is
         declared the depth (slice) axis. Thick axial slices are common
         in clinical brain MRI (e.g. 3–5 mm vs 0.9 mm in-plane).
      2. **Shape heuristic** (fallback):
         If voxel sizes are isotropic (identity affine or truly isotropic
         acquisition), use the axis with the fewest voxels as depth.
         For typical brain MRI the slice count is smaller than the in-plane size.
      3. **Header orientation** (tiebreaker):
         When the two heuristics above are inconclusive, use nibabel's
         orientation codes to find the Superior (I→S) array axis.

    After axis detection the volume is transposed to (D, H, W) = (depth,
    height, width), normalized robustly (1st–99th percentile), and resized
    to ``target_size`` via trilinear interpolation.

    Args:
        nifti_path: Path to NIfTI file
        target_size: Target dimensions (depth, height, width)
        verbose: Whether to print detection details
        
    Returns:
        Volume tensor (1,1,D,H,W) float32 on CPU
    """
    img = nib.load(nifti_path)
    orig_shape = img.shape
    orig_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    # ── 1. Canonical reorientation ──────────────────────────────────────────
    img_can = nib.as_closest_canonical(img)
    vol = img_can.get_fdata(dtype=np.float32)
    can_zooms = tuple(float(z) for z in img_can.header.get_zooms()[:3])
    can_shape = vol.shape

    if verbose:
        print(
            f"[load_nifti_volume_general] '{os.path.basename(nifti_path)}'\n"
            f"  orig shape={orig_shape}  zooms={orig_zooms}\n"
            f"  canonical shape={can_shape}  zooms={can_zooms}"
        )

    # ── 2. Depth-axis detection ──────────────────────────────────────────────
    zooms = np.array(can_zooms, dtype=np.float32)
    shape = np.array(can_shape, dtype=np.int64)

    # Heuristic A: thick-slice detection via voxel size
    median_zoom = float(np.median(zooms))
    zoom_ratios = zooms / (median_zoom + 1e-6)
    thick_axes = np.where(zoom_ratios >= 1.5)[0]

    # Heuristic B: smallest dimension (works when affine is identity/isotropic)
    min_dim_ax = int(np.argmin(shape))

    # Heuristic C: header orientation → find array axis coding for S (axial)
    ornt = nib.orientations.io_orientation(img_can.affine)
    header_depth_ax = None
    for arr_ax, (can_code, _flip) in enumerate(ornt):
        if int(can_code) == 2:  # 2 = Superior direction
            header_depth_ax = arr_ax
            break

    # Choose depth axis: thick axis wins; then smallest-dim; then header
    if len(thick_axes) == 1:
        ax_depth = int(thick_axes[0])
        reason = f"voxel-size heuristic (zoom ratio={zoom_ratios[ax_depth]:.2f})"
    elif len(thick_axes) > 1:
        # Multiple thick axes: pick the one that is also smallest
        candidates = [a for a in thick_axes if shape[a] == shape[thick_axes].min()]
        ax_depth = int(candidates[0]) if candidates else int(thick_axes[0])
        reason = "voxel-size heuristic (multiple thick axes, using smallest)"
    else:
        # Isotropic/identity affine → use smallest dimension
        ax_depth = min_dim_ax
        reason = f"shape heuristic (smallest dim at axis {ax_depth})"
        # If shape is also ambiguous (two equal-smallest dims), let header break tie
        sorted_dims = np.sort(shape)
        if sorted_dims[0] == sorted_dims[1] and header_depth_ax is not None:
            ax_depth = header_depth_ax
            reason += f" → header tiebreaker → axis {ax_depth}"

    if verbose:
        print(f"  depth axis={ax_depth}  ({reason})")

    # ── 3. Assign height and width axes ─────────────────────────────────────
    spatial_axes = [0, 1, 2]
    spatial_axes.remove(ax_depth)

    # Within the two remaining axes, try to align to P-A (height) vs L-R (width)
    # using the orientation codes (1=A, 0=R) so coronal view makes sense.
    code_of = {int(ornt[a, 0]): a for a in spatial_axes}
    ax_height = code_of.get(1, spatial_axes[0])   # A axis → height
    ax_width  = code_of.get(0, spatial_axes[1])   # R axis → width
    # If the codes don't resolve (e.g. identity affine may have no code 1 in
    # the spatial_axes), just keep original order.
    if ax_height == ax_width:
        ax_height, ax_width = spatial_axes[0], spatial_axes[1]

    # ── 4. Transpose to (D, H, W) ───────────────────────────────────────────
    transpose_order = (ax_depth, ax_height, ax_width)
    vol = np.transpose(vol, transpose_order)

    if verbose:
        print(f"  transpose {transpose_order} → shape {vol.shape}")

    # ── 5. Robust normalization ─────────────────────────────────────────────
    vmin, vmax = float(np.percentile(vol, 1)), float(np.percentile(vol, 99))
    if vmax > vmin:
        vol = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        vol = np.zeros_like(vol)

    # ── 6. Resize via trilinear interpolation ───────────────────────────────
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    D, H, W = target_size
    vol_t = F.interpolate(vol_t, size=(D, H, W), mode="trilinear", align_corners=False)

    return vol_t.contiguous().float()


def get_volume_from_ex(ex: Dict, target_size=(64, 128, 128), device=None):
    """
    Load the volume only when needed.
    
    Args:
        ex: Example dictionary containing 'image_path'
        target_size: Target dimensions (depth, height, width)
        device: Target device (CPU if None)
        
    Returns:
        Volume tensor (1,1,D,H,W) float32 on specified device
    """
    vol = load_nifti_volume(ex["image_path"], target_size=target_size)
    if device is not None:
        vol = vol.to(device)
    return vol


# ============================================================
# MODEL ARCHITECTURE - 3D Vision Encoder
# ============================================================

def inflate_conv2d_to_conv3d(conv2d: nn.Conv2d, depth: int, in_channels_override=None) -> nn.Conv3d:
    """
    Inflate a Conv2D layer to Conv3D (kernel_depth = depth) by copying weights.
    
    The 2D kernel is repeated along the depth dimension and averaged to preserve
    the magnitude of activations.
    
    Args:
        conv2d: Source 2D convolution layer
        depth: Kernel size in the depth dimension
        in_channels_override: Override input channels (e.g., 1 for grayscale MRI vs 3 for RGB)
        
    Returns:
        Inflated 3D convolution layer
    """
    padding_2d = conv2d.padding
    if isinstance(padding_2d, str):
        assert padding_2d == "valid", "Only padding='valid' is supported"
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding_2d

    in_channels = in_channels_override if in_channels_override is not None else conv2d.in_channels

    conv3d = nn.Conv3d(
        in_channels=in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=(depth, conv2d.kernel_size[0], conv2d.kernel_size[1]),
        stride=(1, conv2d.stride[0], conv2d.stride[1]),
        padding=(depth // 2, pad_h, pad_w),
        bias=(conv2d.bias is not None),
    )

    with torch.no_grad():
        w2 = conv2d.weight.data  # (out, in, kh, kw)
        # Repeat along depth dimension and average to preserve magnitude
        w3 = w2.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth  # (out, in, kd, kh, kw)

        if in_channels_override is not None and in_channels_override != conv2d.in_channels:
            # Handle channel adaptation (e.g., 1-channel MRI from 3-channel RGB weights)
            if conv2d.in_channels == 3 and in_channels_override == 1:
                w3 = w3.mean(dim=1, keepdim=True)
            else:
                raise ValueError("Channel override not generically handled.")
        conv3d.weight.copy_(w3)

        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.data)

    return conv3d


class SiglipVisionTransformer3D(nn.Module):
    """
    Adapt SigLIP/MedSigLIP vision_model to 3D inputs:
    - patch_embedding → patch_embedding_3d (inflated Conv3D)
    - 3D positional embedding (custom) as in the notebook
    
    The model processes 3D volumes by:
    1. Extracting 3D patches via inflated convolution
    2. Adding learned 3D positional embeddings
    3. Processing through the original 2D transformer encoder
    """
    def __init__(self, vision_model_2d, depth: int = 2, max_depth_patches: int = 128):
        """
        Args:
            vision_model_2d: Pre-trained 2D vision model (SigLIP/MedSigLIP)
            depth: Kernel depth for patch embedding inflation
            max_depth_patches: Maximum number of patches along depth dimension
        """
        super().__init__()
        self.vision_model_2d = vision_model_2d
        self.depth = depth
        self.max_depth_patches = max_depth_patches

        # Inflate 2D patch embedding to 3D
        pe2d = vision_model_2d.embeddings.patch_embedding
        self.patch_embedding_3d = inflate_conv2d_to_conv3d(pe2d, depth=depth, in_channels_override=1)

        # Reuse encoder and normalization layers from 2D model
        self.encoder = vision_model_2d.encoder
        self.post_layernorm = vision_model_2d.post_layernorm

        # Embedding dimension
        self.hidden_size = pe2d.out_channels

        # 2D positional embedding: (1, 1+N, E) often includes cls token
        # In MedSigLIP, cls token is not always used; we sum pos on patch tokens
        self.pos_embed_2d = getattr(vision_model_2d.embeddings, "position_embedding", None)

        # Cache for 3D positional embeddings
        self._pos_cache = {}

    def get_position_embedding_3d(self, Dp: int, num_spatial: int, Hp: int, Wp: int) -> torch.Tensor:
        """
        Create 3D position embedding (Dp * Hp*Wp, E).
        Follows notebook approach: pos = pos_depth + pos_spatial.
        
        Args:
            Dp: Number of patches along depth
            num_spatial: Total number of spatial patches (Hp * Wp)
            Hp: Number of patches along height
            Wp: Number of patches along width
            
        Returns:
            Position embedding tensor (1, N, E) where N = Dp * Hp * Wp
        """
        key = (Dp, Hp, Wp, num_spatial, self.hidden_size)
        if key in self._pos_cache:
            return self._pos_cache[key]

        # Depth positional encoding: (Dp, E)
        pos_d = torch.linspace(-1, 1, steps=Dp).unsqueeze(1)  # (Dp, 1)
        pos_d = pos_d.repeat(1, self.hidden_size)             # (Dp, E)

        # Spatial positional encoding: (Hp*Wp, E)
        ys = torch.linspace(-1, 1, steps=Hp)
        xs = torch.linspace(-1, 1, steps=Wp)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # (Hp*Wp, 2)

        # Simple lifting to embedding dimension
        pos_s = torch.zeros((Hp * Wp, self.hidden_size))
        pos_s[:, 0] = grid[:, 0]
        if self.hidden_size > 1:
            pos_s[:, 1] = grid[:, 1]

        # Broadcast and sum: (Dp, 1, E) + (1, Hp*Wp, E) → (Dp, Hp*Wp, E)
        pos = pos_d.unsqueeze(1) + pos_s.unsqueeze(0)
        pos = pos.reshape(Dp * Hp * Wp, self.hidden_size)  # (N, E)

        pos = pos.unsqueeze(0)  # (1, N, E)
        self._pos_cache[key] = pos
        return pos

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 3D vision transformer.
        
        Args:
            x_3d: Input volume (B, 1, D, H, W)
            
        Returns:
            Encoded patch tokens (B, N, E) where N is number of 3D patches
        """
        # Extract 3D patches
        x = self.patch_embedding_3d(x_3d)          # (B, E, D', H', W')
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, N, E) where N = Dp * Hp * Wp

        # Add 3D positional embedding
        pos = self.get_position_embedding_3d(Dp, Hp * Wp, Hp, Wp).to(x.device, dtype=x.dtype)
        x = x + pos

        # Process through transformer encoder
        out = self.encoder(x).last_hidden_state
        out = self.post_layernorm(out)
        return out


class MedSigLIP3D(nn.Module):
    """
    Wrapper class: load base SigLIP/MedSigLIP (LOCAL) and create 3D vision transformer.
    
    Maintains compatibility with the original 2D model structure while adding 3D capabilities.
    """
    def __init__(self, model_name_or_dir: str, depth: int = 2, max_depth_patches: int = 128):
        """
        Args:
            model_name_or_dir: Path to local SigLIP model directory
            depth: Kernel depth for patch embedding inflation
            max_depth_patches: Maximum number of patches along depth dimension
        """
        super().__init__()
        base = load_sigclip_local(model_name_or_dir)

        # Preserve text model and logit scale (for potential contrastive training)
        self.text_model = getattr(base, "text_model", None)
        self.logit_scale = getattr(base, "logit_scale", None)

        # Create 3D vision model from 2D base
        vision_2d = base.vision_model
        self.vision_model = SiglipVisionTransformer3D(vision_2d, depth=depth, max_depth_patches=max_depth_patches)

    def encode_image(self, x_3d: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D image volume to patch tokens.
        
        Args:
            x_3d: Input volume (B, 1, D, H, W)
            
        Returns:
            Normalized patch tokens (B, N, E) as in SigLIP
        """
        return self.vision_model(x_3d)


class BrainGemma3D(nn.Module):
    """
    Complete system for Report Generation from 3D volumes.
    
    Architecture:
    - Vision: MedSigLIP3D (inflated from 2D medical model)
    - Language: MedGemma CausalLM (quantized)
    - Projection: Linear layers to map vision features to language embedding space
    
    The model concatenates vision tokens with text prompt embeddings and
    uses the language model to generate radiology reports.
    """
    def __init__(
        self,
        vision_model_dir: str,
        language_model_dir: str,
        depth: int = 2,
        max_depth_patches: int = 128,
        num_vision_tokens: int = 32,
        freeze_vision: bool = False,
        freeze_language: bool = True,
        device_map=None,
    ):
        """
        Args:
            vision_model_dir: Path to local vision model directory
            language_model_dir: Path to local language model directory
            depth: Kernel depth for patch embedding inflation
            max_depth_patches: Maximum number of patches along depth dimension
            num_vision_tokens: Number of vision tokens to pass to language model
            freeze_vision: Whether to freeze vision encoder during training
            freeze_language: Whether to freeze language model during training
            device_map: Device mapping for model placement
        """
        super().__init__()

        self.num_vision_tokens = num_vision_tokens
        # Vision scale parameter (lowered from 8.0 to avoid saturation)
        self.vis_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Load language model and tokenizer (LOCAL)
        self.language_model = load_medgemma_lm_local(language_model_dir, device_map=device_map)
        self.tokenizer = load_medgemma_tokenizer_local(language_model_dir)

        self.lm_device = next(self.language_model.parameters()).device
        print(f"[BrainGemma3D] language_model device = {self.lm_device}")

        # Load vision encoder (LOCAL)
        self.vision_encoder = MedSigLIP3D(
            model_name_or_dir=vision_model_dir,
            depth=depth,
            max_depth_patches=max_depth_patches,
        ).to(self.lm_device)

        # Vision-to-language projector
        vision_dim = 1152  # MedSigLIP embedding dimension (from notebook)
        if hasattr(self.language_model.config, "text_config"):
            language_dim = self.language_model.config.text_config.hidden_size
        elif hasattr(self.language_model.config, "hidden_size"):
            language_dim = self.language_model.config.hidden_size
        else:
            language_dim = self.language_model.model.embed_tokens.embedding_dim

        self.vision_projector = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        ).to(self.lm_device, dtype=torch.bfloat16)

        # Initialize projector with Xavier uniform (gain=0.5 for stability)
        for module in self.vision_projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Convert vision encoder to bfloat16 (CRITICAL: prevents type mismatch)
        print(f"[BrainGemma3D] Converting vision encoder to bfloat16...")
        with torch.no_grad():
            self.vision_encoder.vision_model.patch_embedding_3d.weight.data = \
                self.vision_encoder.vision_model.patch_embedding_3d.weight.data.to(torch.bfloat16)
            if self.vision_encoder.vision_model.patch_embedding_3d.bias is not None:
                self.vision_encoder.vision_model.patch_embedding_3d.bias.data = \
                    self.vision_encoder.vision_model.patch_embedding_3d.bias.data.to(torch.bfloat16)
        
        self.vision_encoder = self.vision_encoder.to(dtype=torch.bfloat16)
        print(f"[BrainGemma3D] Vision encoder converted to bfloat16")

        # Apply freezing options
        if freeze_language:
            self.language_model.eval()
            for p in self.language_model.parameters():
                p.requires_grad = False

        if freeze_vision:
            self.vision_encoder.eval()
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # Fix pad token (Gemma often has None)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def encode_volume(self, volume_3d: torch.Tensor) -> torch.Tensor:
        """
        Encode 3D volume to vision tokens projected to language embedding space.
        
        Process:
        1. Extract patch tokens via vision encoder
        2. Pool to fixed number of tokens via adaptive pooling
        3. Project to language embedding dimension
        4. Scale by learnable parameter
        
        Args:
            volume_3d: Input volume (B, 1, D, H, W)
            
        Returns:
            Vision tokens (B, K, H_lm) normalized/scaled for language model
        """
        volume_3d = volume_3d.to(self.lm_device, dtype=torch.bfloat16)

        # Extract patch tokens (B, N, E_vision)
        patch_tokens = self.vision_encoder.encode_image(volume_3d)
        
        # Pool to fixed number of vision tokens
        x = patch_tokens.transpose(1, 2)                            # (B, E_vision, N)
        x = F.adaptive_avg_pool1d(x, self.num_vision_tokens)
        x = x.transpose(1, 2)                                       # (B, K, E_vision)

        # Project to language embedding space and scale
        x = self.vision_projector(x.to(torch.bfloat16))             # (B, K, H_lm)
        x = x * self.vis_scale.to(x.dtype)
        return x

    @torch.no_grad()
    def generate_report(
        self,
        volume_3d: torch.Tensor,
        prompt: str = None,
        max_new_tokens: int = 128,
        min_new_tokens: int = 10,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """
        Generate radiology report from 3D volume.
        
        Concatenates [vision_tokens] + [prompt tokens] as inputs_embeds.
        Includes controls for repetitions/hallucinations via repetition_penalty
        and no_repeat_ngram_size.
        
        Args:
            volume_3d: MRI volume (B, 1, D, H, W)
            prompt: Input prompt (None = use CANONICAL_PROMPT)
            max_new_tokens: Maximum number of tokens to generate
            min_new_tokens: Minimum number of tokens to generate (prevents empty outputs)
            temperature: Sampling temperature (None or 0 = greedy)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for token repetition
            no_repeat_ngram_size: Prevent repetition of n-grams
            
        Returns:
            Generated radiology report text
        """
        if prompt is None:
            prompt = CANONICAL_PROMPT
        
        self.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Encode volume to vision tokens
        vis = self.encode_volume(volume_3d)  # (1, K, H)

        # Tokenize prompt
        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=256,
        ).to(self.lm_device)

        # Get text embeddings
        text_emb = self.language_model.get_input_embeddings()(tok.input_ids)  # (1, T, H)

        # Concatenate vision and text embeddings
        inputs_embeds = torch.cat([vis, text_emb], dim=1)  # (1, K+T, H)
        attn = torch.ones(inputs_embeds.shape[:2], device=self.lm_device, dtype=torch.long)

        # Generate report
        out_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=(temperature is not None and temperature > 0),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=False,  # Don't stop too early (prevents empty outputs)
        )

        # Decode: includes prompt; naively remove prompt by searching first occurrence
        txt = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # Often txt contains prompt + output; try to truncate after prompt
        if prompt.strip() in txt:
            txt = txt.split(prompt.strip(), 1)[-1].strip()
        return txt
