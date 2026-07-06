#!/usr/bin/env python3
"""
BrainGemma3D Inference Script (Fixed Version)
Loads a trained model and generates reports for 3D MRI volumes.

APPLIED FIXES:
- Default prompt now aligned with training (CANONICAL_PROMPT)
- Correct handling of newline (\n) from CLI
- Compatibility with robust load_nifti_volume
"""

import os
import torch
import csv
import sys
from pathlib import Path
from typing import List, Dict

# Import required functions from the new separate files
try:
    from braingemma3d_architecture import (
        BrainGemma3D,
        load_nifti_volume,
        CANONICAL_PROMPT,
    )
    from braingemma3d_training import (
        set_seed,
        save_volume_slices,
        build_balanced_dataset,
        make_group_split,
    )
except ImportError as e:
    print(f"❌ Critical error: Unable to import required modules: {e}")
    print("   Ensure braingemma3d_architecture.py and braingemma3d_training.py are in the same directory.")
    sys.exit(1)


def load_trained_model(
    checkpoint_dir: str,
    vision_model_dir: str,
    language_model_dir: str,
    depth: int = 2,
    max_depth_patches: int = 128,
    num_vision_tokens: int = 32,
    device_map=None,
) -> BrainGemma3D:
    """
    Load the BrainGemma3D model with trained weights (Projector + LoRA)
    """
    print(f"📥 Loading model from {checkpoint_dir}")
    
    # 1) Create base model (architecture must match training)
    try:
        model = BrainGemma3D(
            vision_model_dir=vision_model_dir,
            language_model_dir=language_model_dir,
            depth=depth,
            max_depth_patches=max_depth_patches,
            num_vision_tokens=num_vision_tokens,
            freeze_vision=True,   # Inference: everything frozen
            freeze_language=True,
            device_map=device_map,
        )
    except Exception as e:
        print(f"❌ Error initializing base model: {e}")
        print(f"   Check paths: {vision_model_dir}, {language_model_dir}")
        sys.exit(1)
    
    # 2) Load projector + vis_scale
    proj_path = os.path.join(checkpoint_dir, "projector_vis_scale.pt")
    if os.path.exists(proj_path):
        try:
            ckpt = torch.load(proj_path, map_location=model.lm_device)
            model.vision_projector.load_state_dict(ckpt["vision_projector"])
            if "vis_scale" in ckpt and ckpt["vis_scale"] is not None:
                # Robust handling scalar vs tensor
                val = ckpt["vis_scale"]
                if isinstance(val, torch.Tensor):
                    model.vis_scale.data = val.to(model.lm_device)
                else:
                    model.vis_scale.data.fill_(val)
            print(f"✅ Loaded projector | vis_scale={model.vis_scale.item():.3f}")
        except Exception as e:
            print(f"❌ Error loading projector: {e}")
    else:
        print(f"⚠️  Projector checkpoint not found at {proj_path}")
        print("    The model may produce random outputs (vision not aligned).")
    
    # 3) Load LoRA adapters (if present)
    lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.exists(lora_dir):
        try:
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(
                model.language_model, 
                lora_dir,
                is_trainable=False
            )
            print(f"✅ Loaded LoRA adapters from {lora_dir}")
        except Exception as e:
            print(f"❌ Error loading LoRA: {e}")
    else:
        print(f"ℹ️  No LoRA adapters found (Running in Phase 2A mode or Base model)")
    
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: BrainGemma3D,
    test_data: List[Dict],
    prompt: str = None, 
    target_size=(155, 240, 240),
    max_new_tokens: int = 160,
    min_new_tokens: int = 10,
    temperature: float = 0.1,
    top_p: float = 0.9,
    # --- ADDED THESE TWO MISSING PARAMETERS ---
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    # ----------------------------------------------
    save_reports: bool = True,
    output_dir: str = "inference_output",
    quick: bool = False,
):
    """
    Runs inference on the test set
    """
    model.eval()
    
    # --- PROMPT FIXING ---
    if prompt is None:
        prompt = CANONICAL_PROMPT
        prompt_desc = "CANONICAL (Default)"
    else:
        prompt = prompt.replace("\\n", "\n")
        prompt_desc = f"CUSTOM: {repr(prompt)}"

    if not quick:
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 70)
        print(f"🔮 INFERENCE ON TEST SET (n={len(test_data)})")
        print("=" * 70)
        print(f"Prompt: {prompt_desc}")
        print(f"Params: max_tokens={max_new_tokens}, temp={temperature}, rep_pen={repetition_penalty}")
        print()
    else:
        print(f"⚡ QUICK MODE: Inference on 1 patient...\n")
    
    results = []
    
    for i, ex in enumerate(test_data, 1):
        patient_id = ex.get("patient_id", f"test_{i}")
        if not quick:
            print(f"[{i}/{len(test_data)}] {patient_id}... ", end="", flush=True)
        else:
            print(f"📁 Processing: {patient_id}")
        
        try:
            # Load volume
            vol = load_nifti_volume(ex["image_path"], target_size=target_size)
            
            # Optional visual debug
            if not quick and i <= 1:
                viz_path = os.path.join(output_dir, f"{patient_id}_input_debug.png")
                save_volume_slices(
                    vol,
                    viz_path,
                    title=f"DEBUG INPUT {patient_id}",
                    is_healthy=bool(ex.get("is_healthy", False)),
                )

            # Generate report PASSING THE PARAMETERS
            generated_report = model.generate_report(
                vol,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,      # Pass to model
                no_repeat_ngram_size=no_repeat_ngram_size,  # Pass to model
            )
            
            results.append({
                "patient_id": patient_id,
                "image_path": ex["image_path"],
                "generated_report": generated_report,
                "ground_truth": ex.get("report", ""),
            })
            
            if not quick:
                preview = generated_report.replace('\n', ' ').strip()[:60]
                print(f"✅ {preview}...")
            else:
                print(f"✅ Done")
            
            if save_reports and not quick:
                report_path = os.path.join(output_dir, f"{patient_id}_generated.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(f"Patient: {patient_id}\n")
                    f.write(f"Image: {ex['image_path']}\n")
                    f.write(f"Prompt Used: {repr(prompt)}\n")
                    f.write(f"\n{'='*60}\nGENERATED REPORT:\n{'='*60}\n\n")
                    f.write(generated_report)
                    f.write(f"\n\n{'='*60}\nGROUND TRUTH:\n{'='*60}\n\n")
                    f.write(ex.get("report", "N/A"))
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # import traceback
            # traceback.print_exc()
            results.append({
                "patient_id": patient_id,
                "image_path": ex["image_path"],
                "generated_report": f"ERROR: {str(e)}",
                "ground_truth": ex.get("report", ""),
            })
    
    if not quick:
        csv_path = os.path.join(output_dir, "inference_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "image_path", "generated_report", "ground_truth"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Inference completed! Results in: {output_dir}")
    else:
        if results and not results[0]["generated_report"].startswith("ERROR"):
            print("\n" + "="*80)
            print(f"🤖 GENERATED REPORT | Patient: {results[0]['patient_id']}")
            print("="*80)
            print(results[0]["generated_report"])
            print("\n" + "-"*80)
            print("📋 GROUND TRUTH (Preview)")
            print("-"*80)
            print(results[0]["ground_truth"][:400] + "...")
            print("="*80)
    
    return results


def evaluate_metrics(results: List[Dict]) -> Dict:
    """
    Compute basic metrics (BLEU/ROUGE) if libraries are available
    """
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except ImportError:
        print("⚠️  Metrics not available. Install: pip install rouge-score nltk")
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    bleu_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for r in results:
        if r['generated_report'].startswith("ERROR"): continue
        gen = r['generated_report']
        ref = r['ground_truth']
        if not ref or not gen: continue
        
        # BLEU-1 (simplified)
        gen_tok = gen.split()
        ref_tok = [ref.split()]
        bleu_scores.append(sentence_bleu(ref_tok, gen_tok, weights=(1,0,0,0)))
        
        # ROUGE-L and ROUGE-2
        scores = scorer.score(ref, gen)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        
        return {
        "bleu1": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
        "n_samples": len(bleu_scores),
    }


def main():
    import argparse

    print("\n" + "="*70)
    print("🧠 BRAINGEMMA3D - INFERENCE")
    print("="*60)

    parser = argparse.ArgumentParser(description="BrainGemma3D Inference")

    # ==================== PATHS ====================
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to trained checkpoint (e.g., ckpt_after_step2B_full)")
    parser.add_argument("--base-dir", type=str, default="/leonardo_work/CESMA_leonardo/CBMS",
                        help="Base directory with Models/ and Datasets/ folders")
    
    # ==================== INPUT MODE ====================
    input_group = parser.add_argument_group('Input Selection')
    input_group.add_argument("--input-volume", type=str, default=None,
                        help="Path to custom NIfTI volume (.nii/.nii.gz)")
    input_group.add_argument("--patient-id", type=str, default=None,
                        help="Specific BraTS patient ID")
    input_group.add_argument("--quick", action="store_true",
                        help="Quick test: 1 random patient, terminal output only")
    input_group.add_argument("--is-healthy", action="store_true",
                        help="Mark custom --input-volume as a healthy control (affects debug visualization)")
    
    # ==================== DATASET ====================
    dataset_group = parser.add_argument_group('Dataset Options')
    dataset_group.add_argument("--num-patients", type=int, default=None, help="Number of BraTS patients")
    dataset_group.add_argument("--num-healthy", type=int, default=99, help="Number of healthy brains")
    dataset_group.add_argument("--healthy-dir", type=str, default="Datasets/HealthyBrains_Preprocessed")
    dataset_group.add_argument("--modality", type=str, default="flair")
    dataset_group.add_argument("--target-size", type=int, nargs=3, default=[64, 128, 128])
    
    # ==================== MODEL ARCHITECTURE ====================
    # MUST MATCH TRAINING
    model_group = parser.add_argument_group('Model Config')
    model_group.add_argument("--num-vision-tokens", type=int, default=32)
    model_group.add_argument("--depth", type=int, default=2)
    model_group.add_argument("--max-depth-patches", type=int, default=128)
    
    # ==================== GENERATION ====================
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (Default: None -> Uses CANONICAL_PROMPT)")
    gen_group.add_argument("--max-new-tokens", type=int, default=160)
    gen_group.add_argument("--min-new-tokens", type=int, default=10,
                        help="Minimum tokens to generate (prevents empty outputs)")
    gen_group.add_argument("--temperature", type=float, default=0.1)
    gen_group.add_argument("--top-p", type=float, default=0.9)
    gen_group.add_argument("--repetition-penalty", type=float, default=1.2)
    gen_group.add_argument("--no-repeat-ngram-size", type=int, default=3)
    
    # ==================== OUTPUT ====================
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--output-dir", type=str, default="inference_output")
    output_group.add_argument("--compute-metrics", action="store_true")
    output_group.add_argument("--save-visualizations", action="store_true", default=True)
    output_group.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Paths resolution
    base_dir = Path(args.base_dir)
    vision_model_path = str(base_dir / "Models" / "siglip")
    language_model_path = str(base_dir / "Models" / "medgemma")
    
    print("\n" + "="*70)
    print("🧠 BRAINGEMMA3D - INFERENCE")
    print("="*70)
    
    model = load_trained_model(
        checkpoint_dir=args.checkpoint_dir,
        vision_model_dir=vision_model_path,
        language_model_dir=language_model_path,
        depth=args.depth,
        max_depth_patches=args.max_depth_patches,
        num_vision_tokens=args.num_vision_tokens,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    
    # CASE 1: Custom Input File
    if args.input_volume:
        if not os.path.exists(args.input_volume):
            print(f"❌ File not found: {args.input_volume}")
            return
        
        # Fake dataset item
        test_data = [{
            "patient_id": "Custom",
            "image_path": args.input_volume,
            "report": "N/A",
            "is_healthy": bool(args.is_healthy),
        }]
        run_inference(model, test_data, prompt=args.prompt, target_size=tuple(args.target_size), 
                      output_dir=args.output_dir, quick=args.quick, 
                      max_new_tokens=args.max_new_tokens, min_new_tokens=args.min_new_tokens,
                      temperature=args.temperature)
        return

    # CASE 2: Dataset Loading
    brats_images_base = str(base_dir / "Datasets" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData")
    brats_reports_base = str(base_dir / "Datasets" / "TextBraTS" / "TextBraTSData")
    if os.path.isabs(args.healthy_dir):
        healthy_dir = args.healthy_dir
    else:
        healthy_dir = str(base_dir / args.healthy_dir)

    print("📦 Loading dataset...")
    dataset = build_balanced_dataset(
        brats_images_base=brats_images_base,
        brats_reports_base=brats_reports_base,
        healthy_brains_base=healthy_dir,
        num_brats_patients=args.num_patients,
        num_healthy_patients=args.num_healthy,
        modality=args.modality,
    )
    
    # Split
    _, _, test_data = make_group_split(dataset, seed=args.seed, train_frac=0.7, val_frac=0.1)

    # Print all test patients
    print("\n📋 List of test patients:")
    for patient in test_data:
        print(f"Patient ID: {patient.get('patient_id', 'Unknown')}")

    # Print healthy patients (if any)
    healthy_patients = [p for p in test_data if "healthy" in p.get('patient_id', '').lower()]
    if healthy_patients:
        print("\n🩺 List of healthy patients:")
        for patient in healthy_patients:
            print(f"Patient ID: {patient.get('patient_id', 'Unknown')}")
    else:
        print("\n🩺 No healthy patient found.")

    
    # Filter patient
    if args.patient_id:
        found = [ex for ex in test_data if ex.get('patient_id') == args.patient_id]
        if not found:
            print(f"❌ Patient {args.patient_id} not found in TEST set.")
            return
        test_data = found

    if args.quick and not args.patient_id:
        import random
        test_data = [random.choice(test_data)]

    # Run
    results = run_inference(
        model, test_data, 
        prompt=args.prompt, 
        target_size=tuple(args.target_size),
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        save_reports=not args.quick,
        output_dir=args.output_dir,
        quick=args.quick
    )

    # Metrics
    if args.compute_metrics and not args.quick:
        print("\n📊 Calculating Metrics...")
        m = evaluate_metrics(results)
        if m:
            print(f"  BLEU-1: {m['bleu1']:.4f}")
            print(f"  ROUGE-2: {m['rouge2']:.4f}")
            print(f"  ROUGE-L: {m['rougeL']:.4f}")

if __name__ == "__main__":
    main()