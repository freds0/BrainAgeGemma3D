#!/usr/bin/env python3
"""
MedGemma Baseline Evaluation Script
====================================
Evaluates ONLY the vanilla MedGemma model (non fine-tuned baseline)
on the same Test Set used for BrainGemma3D.

This script establishes the baseline performance of the pre-trained
model without any fine-tuning on the BraTS dataset.

Approach:
- Multiple 2D slices (64 slices 128x128) encoded in base64
- Chat format with neuroradiological system instruction
- Model: Vanilla MedGemma from Models/medgemma

Metrics calculated:
1. NLG: BLEU-1/4, ROUGE-1/2/L, METEOR, CIDEr, BERTScore
2. Clinical: F1-score for Laterality, Anatomy, Pathology
3. Statistical: 95% CI Bootstrap

Usage:
    python evaluate_medgemma.py \\
        --seed 42 \\
        --num-brats-patients 369 \\
        --output-dir evaluation_results/baseline
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText

# Import report generation functions from baseline script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medgemma3d_pure import (
    load_and_process_nifti,
    encode_image_to_base64,
    INSTRUCTION,
    CANONICAL_PROMPT,
)

# ============================================================================
# DATASET UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_balanced_dataset(
    brats_images_base: str,
    brats_reports_base: str,
    healthy_brains_base: str,
    num_brats_patients: int = 369,
    num_healthy_patients: int = 99,
    modality: str = "flair",
) -> List[Dict]:
    """\n    Build balanced dataset with BraTS pathological cases and healthy controls.\n    """
    dataset = []
    
    # 1) BraTS Pathological
    brats_base = Path(brats_images_base)
    reports_base = Path(brats_reports_base)
    
    patient_folders = sorted([d for d in brats_base.iterdir() if d.is_dir()])
    if num_brats_patients:
        patient_folders = patient_folders[:num_brats_patients]
    
    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        image_file = patient_dir / f"{patient_id}_{modality}.nii"
        
        if not image_file.exists():
            continue
        
        # Search for report in patient subfolder
        report_file = reports_base / patient_id / f"{patient_id}_flair_text.txt"
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report_text = f.read().strip()
        else:
            # Fallback to alternative path without subfolder
            report_file_alt = reports_base / f"{patient_id}_flair_text.txt"
            if report_file_alt.exists():
                with open(report_file_alt, 'r', encoding='utf-8') as f:
                    report_text = f.read().strip()
            else:
                report_text = "No report available."
        
        dataset.append({
            "patient_id": patient_id,
            "image_path": str(image_file),
            "report": report_text,
            "is_healthy": False,
            "modality": modality,
        })
    
    # 2) Healthy Brains
    healthy_base = Path(healthy_brains_base)
    if healthy_base.exists():
        healthy_files = sorted(healthy_base.glob("*.nii.gz"))[:num_healthy_patients]
        
        for hf in healthy_files:
            patient_id = hf.stem.replace("_preprocessed", "")
            
            # Generic report for healthy brains
            report_text = "The brain parenchyma appears within normal limits on this FLAIR MRI."
            
            dataset.append({
                "patient_id": patient_id,
                "image_path": str(hf),
                "report": report_text,
                "is_healthy": True,
                "modality": modality,
            })
    
    return dataset


def make_group_split(
    dataset: List[Dict],
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """\n    Split dataset using group-based splitting to prevent data leakage.\n    """
    random.seed(seed)
    
    # Group by patient_id to prevent leakage
    groups = defaultdict(list)
    for ex in dataset:
        groups[ex["patient_id"]].append(ex)
    
    # Shuffle groups
    group_ids = list(groups.keys())
    random.shuffle(group_ids)
    
    # Split
    n_total = len(group_ids)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    
    train_ids = group_ids[:n_train]
    val_ids = group_ids[n_train:n_train + n_val]
    test_ids = group_ids[n_train + n_val:]
    
    train_data = [ex for gid in train_ids for ex in groups[gid]]
    val_data = [ex for gid in val_ids for ex in groups[gid]]
    test_data = [ex for gid in test_ids for ex in groups[gid]]
    
    return train_data, val_data, test_data


# ============================================================================
NLG_AVAILABLE = True
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    
    for res in ['punkt', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{res}' if res=='punkt' else f'corpora/{res}')
        except LookupError: nltk.download(res, quiet=True)
except ImportError:
    print("⚠️  WARNING: nltk o rouge-score non installati.")
    NLG_AVAILABLE = False

try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False


# ============================================================================
# CLINICAL METRICS
# ============================================================================
class ClinicalMetrics:
    def __init__(self):
        self.categories = {
            "laterality": ["left", "right", "bilateral"],
            "anatomy": ["frontal", "parietal", "temporal", "occipital", "cerebellum", "ventricle", "periventricular"],
            "pathology": ["edema", "necrosis", "enhancement", "compression", "tumor", "mass", "lesion"]
        }

    def extract_entities(self, text: str) -> Dict[str, set]:
        text = text.lower()
        for char in ".,;!?()": text = text.replace(char, " ")
        
        found = {cat: set() for cat in self.categories}
        for cat, keywords in self.categories.items():
            for kw in keywords:
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
            
            if len(ref_set) == 0:
                scores[f"clin_{cat}_f1"] = 1.0 if len(hyp_set) == 0 else 0.0
                continue

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            scores[f"clin_{cat}_f1"] = f1
            
        return scores


# ============================================================================
# NLG METRICS
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
        
        scores['bleu1'] = sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0), smoothing_function=self.smoothing)
        scores['bleu4'] = sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing)
        
        try: scores['meteor'] = meteor_score([ref_tok], hyp_tok)
        except: scores['meteor'] = 0.0
            
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
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
def bootstrap_ci(values: List[float], n_bootstraps=1000, ci=0.95):
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
# BERTSCORE COMPUTATION
# ============================================================================
def compute_bertscore_safe(refs: List[str], hyps: List[str], device='cuda'):
    print("\n🤖 Computing BERTScore (Semantic Similarity)...")
    try:
        from bert_score import score
        torch.cuda.empty_cache()
        P, R, F1 = score(hyps, refs, lang="en", verbose=True, device=device, model_type="roberta-base")
        return F1.numpy()
    except ImportError:
        print("⚠️  bert_score non installato. Salto.")
        return []
    except Exception as e:
        print(f"⚠️  BERTScore error: {e}")
        return []


# ============================================================================
# POST-PROCESSING
# ============================================================================
def clean_report(text: str) -> str:
    """
    Remove common headers added by the model (FINDINGS, IMPRESSION, etc.)
    and return only the report body.
    """
    import re
    
    # Pattern to remove common report headers
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
# BASELINE REPORT GENERATION
# ============================================================================
def generate_baseline_report(
    model,
    processor,
    nifti_path: str,
    num_slices: int = 64,
    target_size: tuple = (128, 128),
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    verbose: bool = False,
) -> str:
    """\n    Generate report using vanilla MedGemma baseline (non-fine-tuned).\n    """
    # 1) Load and preprocess NIfTI
    input_images = load_and_process_nifti(
        nifti_path,
        num_slices=num_slices,
        target_size=target_size,
        verbose=verbose
    )
    
    # 2) Build chat message with base64 encoding
    content = []
    content.append({"type": "text", "text": INSTRUCTION})
    
    for slice_number, img in enumerate(input_images, 1):
        img_base64 = encode_image_to_base64(img, format="jpeg")
        content.append({"type": "image", "image": img_base64})
        content.append({"type": "text", "text": f"SLICE {slice_number}"})
    
    content.append({"type": "text", "text": CANONICAL_PROMPT.strip()})
    
    messages = [{"role": "user", "content": content}]
    
    # 3) Tokenize
    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    
    # 4) Generate
    with torch.inference_mode():
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        
        generation = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
        )
        
        # Post-process
        medgemma_response = processor.post_process_image_text_to_text(
            generation, skip_special_tokens=True
        )[0]
        
        decoded_inputs = processor.post_process_image_text_to_text(
            model_inputs["input_ids"], skip_special_tokens=True
        )[0]
        
        index_input_text = medgemma_response.find(decoded_inputs)
        if 0 <= index_input_text <= 2:
            medgemma_response = medgemma_response[index_input_text + len(decoded_inputs):]
        
        # Clean report by removing FINDINGS/IMPRESSION headers
        decoded = clean_report(medgemma_response.strip())
    
    return decoded


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MedGemma Baseline Evaluation")
    
    # Paths
    parser.add_argument("--model-id", type=str, 
                       default="/leonardo_work/CESMA_leonardo/CBMS/Models/medgemma",
                       help="Path locale al modello MedGemma vanilla")
    parser.add_argument("--base-dir", type=str, default="/leonardo_work/CESMA_leonardo/CBMS")
    parser.add_argument("--output-dir", type=str, default="evaluation_results/baseline")
    
    # Data configuration (MUST BE IDENTICAL TO TRAINING for correct test set)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-brats-patients", type=int, default=369)
    parser.add_argument("--num-healthy-patients", type=int, default=99)
    parser.add_argument("--modality", type=str, default="flair")
    
    # Preprocessing
    parser.add_argument("--num-slices", type=int, default=64)
    parser.add_argument("--target-size", type=int, nargs=2, default=[128, 128])
    
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. DATASET RECONSTRUCTION
    print("\n📚 Reconstructing Dataset Logic (Seed 42)...")
    brats_images = str(Path(args.base_dir) / "Datasets" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData")
    brats_reports = str(Path(args.base_dir) / "Datasets" / "TextBraTS" / "TextBraTSData")
    healthy_brains = str(Path(args.base_dir) / "Datasets" / "HealthyBrains_Preprocessed")
    
    dataset = build_balanced_dataset(
        brats_images_base=brats_images,
        brats_reports_base=brats_reports,
        healthy_brains_base=healthy_brains,
        num_brats_patients=args.num_brats_patients,
        num_healthy_patients=args.num_healthy_patients,
        modality=args.modality,
    )
    
    # 70/10/20 split EXACTLY as in training
    _, _, test_data = make_group_split(dataset, seed=args.seed, train_frac=0.7, val_frac=0.1)
    
    print(f"✅ Test Set: {len(test_data)} patients")
    print(f"   BraTS (Pathological): {sum(1 for x in test_data if not x['is_healthy'])}")
    print(f"   Healthy:              {sum(1 for x in test_data if x['is_healthy'])}")

    # 2. LOAD BASELINE MODEL (vanilla, non fine-tuned)
    print(f"\n📥 Loading MedGemma Baseline from: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_buffers=True,
    )
    model.eval()
    print(f"✅ Model loaded on: {model.device}")

    # 3. GENERATION LOOP
    print(f"\n🚀 Starting Baseline Evaluation on Test Set...")
    results = []
    
    clin_metrics = ClinicalMetrics()
    nlg_calc = NLGMetricsCalculator()
    
    acc_clin = defaultdict(list)
    acc_nlg = defaultdict(list)
    
    all_refs = []
    all_hyps = []
    
    refs_cider = {}
    hyps_cider = {}

    for i, ex in enumerate(tqdm(test_data)):
        try:
            gen_text = generate_baseline_report(
                model=model,
                processor=processor,
                nifti_path=ex["image_path"],
                num_slices=args.num_slices,
                target_size=tuple(args.target_size),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                verbose=False,
            )
            
            gt_text = ex["report"]
            
            all_refs.append(gt_text)
            all_hyps.append(gen_text)
            
            sample_res = {
                "patient_id": ex["patient_id"],
                "type": "healthy" if ex["is_healthy"] else "pathological",
                "gt": gt_text,
                "gen": gen_text
            }
            
            # Clinical
            c_scores = clin_metrics.compute_scores(gt_text, gen_text)
            sample_res.update(c_scores)
            for k, v in c_scores.items(): acc_clin[k].append(v)
            
            # NLG
            n_scores = nlg_calc.compute_sample(gt_text, gen_text)
            sample_res.update(n_scores)
            for k, v in n_scores.items(): acc_nlg[k].append(v)
            
            # CIDEr
            refs_cider[str(i)] = [gt_text]
            hyps_cider[str(i)] = [gen_text]
            
            results.append(sample_res)
            
        except Exception as e:
            print(f"Error {ex['patient_id']}: {e}")
            import traceback
            traceback.print_exc()

    # 4. BERTSCORE (free model memory first)
    del model
    torch.cuda.empty_cache()
    
    bert_scores = compute_bertscore_safe(all_refs, all_hyps)
    if len(bert_scores) > 0:
        acc_nlg['bertscore'] = bert_scores

    # 5. STATISTICHE FINALI
    print("\n📊 Calculating Final Statistics...")
    
    final_stats = {
        "config": vars(args),
        "model_type": "baseline_vanilla",
        "metrics": {}
    }
    
    combined_acc = {**acc_clin, **acc_nlg}
    
    for k, values in combined_acc.items():
        mean = np.mean(values)
        low, high = bootstrap_ci(values)
        final_stats["metrics"][k] = {
            "mean": float(mean),
            "ci_95": [float(low), float(high)]
        }
    
    # CIDEr
    if nlg_calc.cider_scorer:
        cider = nlg_calc.compute_corpus_cider(refs_cider, hyps_cider)
        final_stats["metrics"]["cider"] = {"mean": cider, "ci_95": [0.0, 0.0]}

    # Save results
    import pandas as pd
    
    out_json = os.path.join(args.output_dir, "metrics_baseline.json")
    with open(out_json, "w") as f:
        json.dump(final_stats, f, indent=2)
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "evaluation_samples_baseline.csv"), index=False)
    
    print(f"💾 Results saved to: {args.output_dir}")

    # 6. PRINT FINAL TABLE
    print("\n" + "="*70)
    print("MEDGEMMA BASELINE (Vanilla) - TEST SET EVALUATION")
    print("="*70)
    print(f"{'METRIC':<25} | {'MEAN':<10} | {'95% CI':<15}")
    print("-" * 70)
    
    metrics_order = [
        "bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "meteor", "bertscore", "cider",
        "clin_laterality_f1", "clin_anatomy_f1", "clin_pathology_f1"
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
