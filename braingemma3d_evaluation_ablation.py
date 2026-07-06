#!/usr/bin/env python3
"""
BrainGemma3D Scientific Evaluation Script
===========================================================
Performs comprehensive evaluation on the "Unseen" Test Set to produce
tables ready for scientific publication.

Metrics calculated:
1. NLG (Text Quality): BLEU-1/4, ROUGE-1/2/L, METEOR, CIDEr, BERTScore
2. Clinical Efficacy (Diagnostic Accuracy): F1-score for Laterality, Anatomy, Pathology
3. Statistics: 95% Confidence Intervals (Bootstrap)

Usage:
    python braingemma3d_evaluation_ablation.py \
        --checkpoint-dir checkpoints/ESPERIMENTO/phase2b_final \
        --num-brats-patients 369 \
        --seed 42 \
        --output-dir results_paper
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from datetime import datetime

# --- MODULE IMPORTS ---
try:
    from braingemma3d_architecture import (
        BrainGemma3D,
        load_nifti_volume,
        CANONICAL_PROMPT,
        get_volume_from_ex,
    )
    from braingemma3d_training import (
        set_seed,
        build_balanced_dataset,
        make_group_split,
    )
except ImportError as e:
    print(f"❌ Critical Error: Unable to import required modules: {e}")
    print("   Make sure braingemma3d_architecture.py and braingemma3d_training.py are in the same directory.")
    sys.exit(1)

# --- METRIC LIBRARIES ---
NLG_AVAILABLE = True
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    
    # Download NLTK resources
    for res in ['punkt', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{res}' if res=='punkt' else f'corpora/{res}')
        except LookupError: nltk.download(res, quiet=True)
except ImportError:
    print("⚠️  WARNING: nltk o rouge-score not installed.")
    NLG_AVAILABLE = False

try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False


# ============================================================================
# 1. CLINICAL METRICS CLASS
# ============================================================================
class ClinicalMetrics:
    def __init__(self):
        self.categories = {
            "laterality": ["left", "right", "bilateral"],
            "anatomy": ["frontal", "parietal", "temporal", "occipital", "cerebellum", "ventricle", "periventricular"],
            "pathology": ["edema", "necrosis", "enhancement", "compression", "tumor", "mass", "lesion"]
        }

    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        text = text.lower()
        # Remove punctuation for clean matching
        for char in ".,;!?()": text = text.replace(char, " ")
        
        found = {cat: set() for cat in self.categories}
        for cat, keywords in self.categories.items():
            for kw in keywords:
                # Match whole word (prevents "low" from matching inside "slow")
                if f" {kw} " in f" {text} ": 
                    found[cat].add(kw)
        return found

    def compute_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        ref_ents = self.extract_entities(reference)
        hyp_ents = self.extract_entities(hypothesis)
        scores = {}
        
        for cat in self.categories:
            ref_set = ref_ents[cat]
            hyp_set = hyp_ents[cat]
            
            tp = len(ref_set.intersection(hyp_set))
            fp = len(hyp_set - ref_set)
            fn = len(ref_set - hyp_set)
            
            # If GT has no entities, F1 is 1.0 if Hyp also has none, 0.0 otherwise
            if len(ref_set) == 0:
                scores[f"clin_{cat}_f1"] = 1.0 if len(hyp_set) == 0 else 0.0
                continue

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            scores[f"clin_{cat}_f1"] = f1
            
        return scores

# ============================================================================
# 2. NLG METRICS CALCULATOR
# ============================================================================
class NLGMetricsCalculator:
    def __init__(self):
        if NLG_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.cider_scorer = Cider() if CIDER_AVAILABLE else None

    def compute_sample(self, ref: str, hyp: str) -> Dict[str, float]:
        if not NLG_AVAILABLE: return {}
        
        scores = {}
        ref_tok = ref.lower().split()
        hyp_tok = hyp.lower().split()
        
        # BLEU
        scores['bleu1'] = sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
        scores['bleu4'] = sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
        
        # METEOR
        try: scores['meteor'] = meteor_score([ref_tok], hyp_tok)
        except: scores['meteor'] = 0.0
            
        # ROUGE
        rouge = self.rouge_scorer.score(ref, hyp)
        scores['rouge1'] = rouge['rouge1'].fmeasure
        scores['rouge2'] = rouge['rouge2'].fmeasure
        scores['rougeL'] = rouge['rougeL'].fmeasure
        
        return scores

    def compute_corpus_cider(self, refs: Dict[str, List[str]], hyps: Dict[str, List[str]]) -> float:
        if self.cider_scorer:
            try:
                score, _ = self.cider_scorer.compute_score(refs, hyps)
                return score
            except: return 0.0
        return 0.0

# ============================================================================
# 3. BOOTSTRAP STATISTICS (95% CI)
# ============================================================================
def bootstrap_ci(values: List[float], n_bootstraps=1000, ci=0.95):
    """Compute confidence interval using bootstrap resampling"""
    data = np.array(values)
    if len(data) < 2: return 0.0, 0.0
    
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

# ============================================================================
# 4. BERTSCORE FUNCTION (Optional but powerful)
# ============================================================================
def compute_bertscore_safe(refs: List[str], hyps: List[str], device='cuda'):
    print("\n🤖 Computing BERTScore (Semantic Similarity)...")
    try:
        from bert_score import score
        # Free GPU memory before loading another model
        torch.cuda.empty_cache()
        P, R, F1 = score(hyps, refs, lang="en", verbose=True, device=device, model_type="roberta-base")
        return F1.numpy()
    except ImportError:
        print("⚠️  bert_score not installed. Skipping.")
        return []
    except Exception as e:
        print(f"⚠️  BERTScore error (often insufficient memory): {e}")
        return []


# ============================================================================
# 5. MODEL LOADING
# ============================================================================
def load_eval_model(args):
    print(f"\n📥 Loading model from: {args.checkpoint_dir}")
    
    # Check Prompt
    if args.prompt is None:
        args.prompt = CANONICAL_PROMPT
    else:
        args.prompt = args.prompt.replace("\\n", "\n")

    # Initialize base model
    model = BrainGemma3D(
        vision_model_dir=str(Path(args.base_dir) / "Models" / "siglip"),
        language_model_dir=str(Path(args.base_dir) / "Models" / "medgemma"),
        depth=args.depth,
        num_vision_tokens=args.num_vision_tokens,
        freeze_vision=True,
        freeze_language=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    
    # Load projector weights
    proj_path = os.path.join(args.checkpoint_dir, "projector_vis_scale.pt")
    if os.path.exists(proj_path):
        ckpt = torch.load(proj_path, map_location=model.lm_device)
        model.vision_projector.load_state_dict(ckpt["vision_projector"])
        if "vis_scale" in ckpt and ckpt["vis_scale"] is not None:
            val = ckpt["vis_scale"]
            if isinstance(val, torch.Tensor): model.vis_scale.data = val.to(model.lm_device)
            else: model.vis_scale.data.fill_(val)
        print(f"✅ Loaded Projector (vis_scale={model.vis_scale.item():.3f})")
    else:
        print(f"❌ ERROR: Projector file not found in {proj_path}")
        sys.exit(1)

    # Load LoRA adapters (optional - only for phase2a and phase2b)
    lora_dir = os.path.join(args.checkpoint_dir, "lora_adapters")
    lora_loaded = False
    if os.path.exists(lora_dir):
        try:
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(model.language_model, lora_dir)
            print("✅ Loaded LoRA Adapters")
            lora_loaded = True
        except Exception as e:
            print(f"⚠️  Warning: LoRA adapters found but failed to load: {e}")
            print("   Continuing with base model + projector only (Phase 1 mode)")
    else:
        print("ℹ️  No LoRA adapters found - evaluating Phase 1 (Vision Projector only)")
    
    model.eval()
    model._lora_loaded = lora_loaded  # Store for reporting
    return model


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="BrainGemma3D Scientific Evaluation")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, required=True, 
                        help="Path to checkpoint directory (e.g., checkpoints/exp/phase1_alignment)")
    parser.add_argument("--base-dir", type=str, default="/leonardo_work/CESMA_leonardo/CBMS")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--phase-name", type=str, default=None, 
                        help="Phase name for ablation study (e.g., 'phase1', 'phase2a', 'phase2b_final')")
    
    # Data split configuration - MUST BE IDENTICAL TO TRAINING FOR CORRECT TEST SET
    parser.add_argument("--seed", type=int, default=42, help="Seed 42 is critical for correct split")
    parser.add_argument("--num-brats-patients", type=int, default=369) # Default used in training
    parser.add_argument("--num-healthy-patients", type=int, default=99) # Default used in training
    parser.add_argument("--modality", type=str, default="flair")
    
    # Model Params
    parser.add_argument("--num-vision-tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--target-size", type=int, nargs=3, default=[64, 128, 128])
    
    # Generation parameters (conservative for benchmarking)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. DATASET RECONSTRUCTION & SPLIT (La parte fondamentale)
    print("\n📚 Reconstructing Dataset Logic (Seed 42)...")
    brats_images = str(Path(args.base_dir) / "Datasets" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData")
    brats_reports = str(Path(args.base_dir) / "Datasets" / "TextBraTS" / "TextBraTSData")
    healthy_brains = str(Path(args.base_dir) / "Datasets" / "HealthyBrains_Preprocessed")
    
    # Reconstruct entire dataset with "balanced" logic
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

    # 2. LOAD MODEL
    model = load_eval_model(args)
    
    # 3. GENERATION LOOP
    print(f"\n🚀 Starting Generation on Test Set...")
    results = []
    
    clin_metrics = ClinicalMetrics()
    nlg_calc = NLGMetricsCalculator()
    
    acc_clin = defaultdict(list)
    acc_nlg = defaultdict(list)
    
    all_refs = []
    all_hyps = []
    
    # Dictionaries for CIDEr (requires specific structure)
    refs_cider = {}
    hyps_cider = {}

    for i, ex in enumerate(tqdm(test_data)):
        try:
            # Load volume using robust function from training module
            vol = load_nifti_volume(ex["image_path"], target_size=tuple(args.target_size))
            vol = vol.to(model.lm_device)
            if vol.ndim == 4: vol = vol.unsqueeze(0) # Batch dim
            
            # Generazione
            with torch.no_grad():
                gen_text = model.generate_report(
                    vol,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
            
            gt_text = ex["report"]
            
            # Accumulate for global BERTScore
            all_refs.append(gt_text)
            all_hyps.append(gen_text)
            
            # Calculate metrics for this sample
            sample_res = {
                "patient_id": ex["patient_id"],
                "type": "healthy" if ex["is_healthy"] else "pathological",
                "gt": gt_text,
                "gen": gen_text
            }
            
            # Clinical Metrics
            c_scores = clin_metrics.compute_scores(gt_text, gen_text)
            sample_res.update(c_scores)
            for k, v in c_scores.items(): acc_clin[k].append(v)
            
            # NLG Metrics
            n_scores = nlg_calc.compute_sample(gt_text, gen_text)
            sample_res.update(n_scores)
            for k, v in n_scores.items(): acc_nlg[k].append(v)
            
            # CIDEr Prep
            refs_cider[str(i)] = [gt_text]
            hyps_cider[str(i)] = [gen_text]
            
            results.append(sample_res)
            
        except Exception as e:
            print(f"Error {ex['patient_id']}: {e}")

    # 4. CALCULATE BERTSCORE (Global)
    # Save model info before deleting
    lora_was_loaded = getattr(model, '_lora_loaded', False)
    
    # Free model memory to make room for BERT
    del model
    torch.cuda.empty_cache()
    
    bert_scores = compute_bertscore_safe(all_refs, all_hyps)
    if len(bert_scores) > 0:
        acc_nlg['bertscore'] = bert_scores

    # 5. FINAL STATISTICS AND SAVING
    print("\n📊 Calculating Final Statistics...")
    
    # Determine model phase
    phase_info = {
        "checkpoint_dir": args.checkpoint_dir,
        "lora_loaded": lora_was_loaded,
        "phase_name": args.phase_name or "unknown"
    }
    
    final_stats = {
        "config": vars(args), 
        "phase_info": phase_info,
        "metrics": {}
    }
    
    # Combine all results
    combined_acc = {**acc_clin, **acc_nlg}
    
    for k, values in combined_acc.items():
        mean = np.mean(values)
        low, high = bootstrap_ci(values)
        final_stats["metrics"][k] = {
            "mean": float(mean),
            "ci_95": [float(low), float(high)]
        }
    
    # CIDEr (Corpus Level)
    if nlg_calc.cider_scorer:
        cider = nlg_calc.compute_corpus_cider(refs_cider, hyps_cider)
        final_stats["metrics"]["cider"] = {"mean": cider, "ci_95": [0.0, 0.0]}

    # Save JSON
    out_json = os.path.join(args.output_dir, "metrics_full.json")
    with open(out_json, "w") as f:
        json.dump(final_stats, f, indent=2)
        
    # Save detailed CSV for qualitative analysis
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "evaluation_samples.csv"), index=False)
    
    print(f"💾 Results saved to: {args.output_dir}")

    # 6. PRINT FINAL TABLE (Copy-Paste for Paper)
    print("\n" + "="*70)
    print(f"ABLATION STUDY - Phase: {phase_info['phase_name']}")
    print(f"LoRA Loaded: {phase_info['lora_loaded']}")
    print("="*70)
    print(f"{'METRIC':<25} | {'MEAN':<10} | {'95% CI':<15}")
    print("-" * 70)
    
    # Logical order for publication
    metrics_order = [
        "bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "meteor", "bertscore", "cider", # NLG
        "clin_laterality_f1", "clin_anatomy_f1", "clin_pathology_f1" # CLINICAL
    ]
    
    for m in metrics_order:
        if m in final_stats["metrics"]:
            d = final_stats["metrics"][m]
            mean_val = d['mean']
            ci_low = d['ci_95'][0]
            ci_high = d['ci_95'][1]
            print(f"{m.upper():<25} | {mean_val:.4f}     | [{ci_low:.3f}, {ci_high:.3f}]")
    print("="*70)

if __name__ == "__main__":
    main()
