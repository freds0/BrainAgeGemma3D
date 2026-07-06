#!/usr/bin/env python3
"""
Simple evaluator for MD3VLM-generated reports.
Loads generated reports and ground-truth TextBraTS reports and computes
basic NLG + clinical metrics compatible with `evaluate_baseline.py`.

Usage:
    python Baseline/evaluate_md3vlm.py --generated-dir <dir> --ground-truth-dir <dir>
"""

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm
import sys
import torch
from collections import defaultdict

# Minimal metric imports (optional dependencies)
NLG_AVAILABLE = True
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    for res in ['punkt', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{res}' if res=='punkt' else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
except Exception:
    NLG_AVAILABLE = False

try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except Exception:
    CIDER_AVAILABLE = False

# Clinical metrics class with lightweight entity extraction
class ClinicalMetrics:
    def __init__(self):
        self.categories = {
            "laterality": ["left", "right", "bilateral"],
            "anatomy": ["frontal", "parietal", "temporal", "occipital", "cerebellum", "ventricle", "periventricular"],
            "pathology": ["edema", "necrosis", "enhancement", "compression", "tumor", "mass", "lesion"]
        }

    def extract_entities(self, text: str):
        text = (text or "").lower()
        for ch in ".,;!?()":
            text = text.replace(ch, " ")
        found = {cat: set() for cat in self.categories}
        for cat, kws in self.categories.items():
            for kw in kws:
                if f" {kw} " in f" {text} ":
                    found[cat].add(kw)
        return found

    def compute_scores(self, reference: str, hypothesis: str):
        ref = self.extract_entities(reference)
        hyp = self.extract_entities(hypothesis)
        scores = {}
        for cat in self.categories:
            rset = ref[cat]
            hset = hyp[cat]
            tp = len(rset & hset)
            fp = len(hset - rset)
            fn = len(rset - hset)
            if len(rset) == 0:
                scores[f"clin_{cat}_f1"] = 1.0 if len(hset) == 0 else 0.0
                continue
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            scores[f"clin_{cat}_f1"] = f1
        return scores

# NLG metrics calculator
class NLGMetricsCalculator:
    def __init__(self):
        if NLG_AVAILABLE:
            self.smoothing = SmoothingFunction().method1
            self.rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        self.cider = Cider() if CIDER_AVAILABLE else None

    def compute_corpus_cider(self, refs: Dict[str, list], hyps: Dict[str, list]) -> float:
        """
        Compute corpus-level CIDEr if available. `refs` and `hyps` should be dicts
        mapping string ids to list-of-references and list-of-hypotheses respectively,
        matching the pycocoevalcap API.
        """
        if not self.cider:
            return 0.0
        try:
            score, _ = self.cider.compute_score(refs, hyps)
            return float(score)
        except Exception:
            return 0.0

    def compute_sample(self, ref: str, hyp: str) -> Dict[str, float]:
        if not NLG_AVAILABLE:
            return {}
        r_tok = (ref or "").lower().split()
        h_tok = (hyp or "").lower().split()
        scores = {}
        try:
            scores['bleu1'] = sentence_bleu([r_tok], h_tok, weights=(1,0,0,0), smoothing_function=self.smoothing)
            scores['bleu4'] = sentence_bleu([r_tok], h_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=self.smoothing)
        except Exception:
            scores['bleu1'] = scores['bleu4'] = 0.0
        try:
            scores['meteor'] = meteor_score([r_tok], h_tok)
        except Exception:
            scores['meteor'] = 0.0
        try:
            rouge_res = self.rouge.score(ref or "", hyp or "")
            scores['rouge1'] = rouge_res['rouge1'].fmeasure
            scores['rouge2'] = rouge_res['rouge2'].fmeasure
            scores['rougeL'] = rouge_res['rougeL'].fmeasure
        except Exception:
            scores['rouge1'] = scores['rouge2'] = scores['rougeL'] = 0.0
        return scores

# Utility functions for loading reports

def load_generated_reports(output_dir: str) -> Dict[str, str]:
    out = {}
    p = Path(output_dir)
    if not p.exists():
        return out
    # try JSON summary
    s = p / 'generation_summary.json'
    if s.exists():
        try:
            with open(s,'r') as f:
                return json.load(f)
        except Exception:
            pass
    for f in sorted(p.glob('*_generated.txt')):
        pid = f.stem.replace('_flair_generated','').replace('_generated','')
        try:
            with open(f,'r', encoding='utf-8') as fh:
                out[pid] = fh.read().strip()
        except Exception:
            out[pid] = ''
    return out


def load_ground_truth_reports(reports_base_dir: str, modality: str='flair') -> Dict[str, str]:
    out = {}
    base = Path(reports_base_dir)
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        p = d / f"{pid}_{modality}_text.txt"
        if p.exists():
            try:
                with open(p,'r',encoding='utf-8') as fh:
                    out[pid] = fh.read().strip()
            except Exception:
                out[pid] = ''
    return out


# Evaluation of single report pair

def evaluate_single_pair(reference: str, hypothesis: str) -> Dict[str, float]:
    clin = ClinicalMetrics()
    nlg = NLGMetricsCalculator()
    res = {}
    res.update(clin.compute_scores(reference, hypothesis))
    res.update(nlg.compute_sample(reference, hypothesis))
    res['ref_length'] = len((reference or "").split())
    res['hyp_length'] = len((hypothesis or "").split())
    return res


# -----------------------
# Dataset utilities
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_balanced_dataset(
    brats_images_base: str,
    brats_reports_base: str,
    healthy_brains_base: str,
    num_brats_patients: int = 369,
    num_healthy_patients: int = 99,
    modality: str = "flair",
) -> List[Dict]:
    dataset = []
    brats_base = Path(brats_images_base)
    reports_base = Path(brats_reports_base)

    patient_folders = sorted([d for d in brats_base.iterdir() if d.is_dir()]) if brats_base.exists() else []
    if num_brats_patients:
        patient_folders = patient_folders[:num_brats_patients]

    for patient_dir in patient_folders:
        patient_id = patient_dir.name
        image_file = patient_dir / f"{patient_id}_{modality}.nii"
        if not image_file.exists():
            continue
        report_file = reports_base / patient_id / f"{patient_id}_{modality}_text.txt"
        if report_file.exists():
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_text = f.read().strip()
            except Exception:
                report_text = ''
        else:
            report_file_alt = reports_base / f"{patient_id}_{modality}_text.txt"
            if report_file_alt.exists():
                try:
                    with open(report_file_alt, 'r', encoding='utf-8') as f:
                        report_text = f.read().strip()
                except Exception:
                    report_text = ''
            else:
                report_text = "No report available."

        dataset.append({
            "patient_id": patient_id,
            "image_path": str(image_file),
            "report": report_text,
            "is_healthy": False,
            "modality": modality,
        })

    healthy_base = Path(healthy_brains_base)
    if healthy_base.exists():
        healthy_files = sorted(healthy_base.glob("*.nii.gz"))[:num_healthy_patients]
        for hf in healthy_files:
            patient_id = hf.stem.replace("_preprocessed", "")
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
) -> (List[Dict], List[Dict], List[Dict]):
    random.seed(seed)
    groups = defaultdict(list)
    for ex in dataset:
        groups[ex['patient_id']].append(ex)
    group_ids = list(groups.keys())
    random.shuffle(group_ids)
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


# Bootstrap confidence interval computation
def bootstrap_ci(values, n_bootstraps=1000, ci=0.95):
    import numpy as _np
    data = _np.array(values)
    if len(data) < 2:
        return 0.0, 0.0
    rng = _np.random.RandomState(42)
    means = []
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    lower = float(_np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(_np.percentile(means, (1 + ci) / 2 * 100))
    return lower, upper


def compute_bertscore_safe(refs: List[str], hyps: List[str], device: str = 'cuda'):
    try:
        from bert_score import score
        import torch
        torch.cuda.empty_cache()
        P, R, F1 = score(hyps, refs, lang="en", verbose=False, device=device, model_type="roberta-base")
        return F1.numpy().tolist()
    except Exception:
        return []


# Aggregate evaluation across all reports

def evaluate_baseline(generated_reports: Dict[str,str], ground_truth_reports: Dict[str,str], output_dir: str) -> Dict[str, float]:
    common = set(generated_reports.keys()) & set(ground_truth_reports.keys())
    if len(common) == 0:
        print('No overlapping patient ids between generated and ground truth')
        return {}
    all_metrics = []
    detailed = {}
    all_refs = []
    all_hyps = []
    refs_cider = {}
    hyps_cider = {}
    for pid in tqdm(sorted(common), desc='Eval'):
        ref = ground_truth_reports.get(pid,'')
        hyp = generated_reports.get(pid,'')
        if not hyp:
            continue
        m = evaluate_single_pair(ref, hyp)
        all_metrics.append(m)
        detailed[pid] = {'reference': ref, 'hypothesis': hyp, 'metrics': m}
        all_refs.append(ref)
        all_hyps.append(hyp)
        idx = str(len(all_refs)-1)
        refs_cider[idx] = [ref]
        hyps_cider[idx] = [hyp]
    if len(all_metrics) == 0:
        print('No valid evaluations')
        return {}
    agg = defaultdict(list)
    for m in all_metrics:
        for k,v in m.items():
            if isinstance(v, (int,float)):
                agg[k].append(v)
    # Build final aggregated stats with bootstrap CI
    final_stats = {
        'metrics': {},
        'n_evaluated': len(all_metrics),
        'n_total': len(common)
    }

    for k, vals in agg.items():
        mean = float(np.mean(vals))
        low, high = bootstrap_ci(vals)
        final_stats['metrics'][k] = {'mean': mean, 'ci_95': [float(low), float(high)]}

    # Compute BERTScore per-sample if available
    bert_vals = compute_bertscore_safe(all_refs, all_hyps)
    if len(bert_vals) == len(all_refs) and len(bert_vals) > 0:
        import numpy as _np
        mean = float(_np.mean(bert_vals))
        low, high = bootstrap_ci(bert_vals)
        final_stats['metrics']['bertscore'] = {'mean': mean, 'ci_95': [float(low), float(high)]}

    # Compute CIDEr corpus score if available
    nlg_calc = NLGMetricsCalculator()
    if CIDER_AVAILABLE and nlg_calc.cider is not None:
        cider_score = nlg_calc.compute_corpus_cider(refs_cider, hyps_cider)
        final_stats['metrics']['cider'] = {'mean': float(cider_score), 'ci_95': [0.0, 0.0]}

    os.makedirs(output_dir, exist_ok=True)
    # Save simple metrics file (for backward compat)
    with open(os.path.join(output_dir,'evaluation_metrics.json'),'w') as f:
        json.dump(final_stats, f, indent=2)
    with open(os.path.join(output_dir,'detailed_results.json'),'w') as f:
        json.dump(detailed, f, indent=2)

    # Also save a CSV of per-sample metrics if pandas available
    try:
        import pandas as _pd
        rows = []
        for pid, info in detailed.items():
            row = {'patient_id': pid, 'reference': info['reference'], 'hypothesis': info['hypothesis']}
            row.update(info['metrics'])
            rows.append(row)
        df = _pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir,'evaluation_samples_md3vlm.csv'), index=False)
    except Exception:
        pass

    print(f"Saved results to {output_dir}")
    return final_stats


# CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate MD3VLM on BraTS test set (like evaluate_baseline)")

    # Paths and model
    parser.add_argument("--model-path", type=str, default="/leonardo_work/CESMA_leonardo/CBMS/Models/md3vlm")
    parser.add_argument("--base-dir", type=str, default="/leonardo_work/CESMA_leonardo/CBMS")
    parser.add_argument("--output-dir", type=str, default="evaluation_results/md3vlm")

    # Data split / seed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-brats-patients", type=int, default=369)
    parser.add_argument("--num-healthy-patients", type=int, default=99)
    parser.add_argument("--modality", type=str, default="flair")

    # Preprocessing / generation
    parser.add_argument("--target-size", type=int, nargs=3, default=[128,256,256])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-sampling", action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Reconstruct dataset and split like evaluate_baseline
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

    _, _, test_data = make_group_split(dataset, seed=args.seed, train_frac=0.7, val_frac=0.1)

    print(f"✅ Test Set: {len(test_data)} patients")
    print(f"   BraTS (Pathological): {sum(1 for x in test_data if not x['is_healthy'])}")
    print(f"   Healthy:              {sum(1 for x in test_data if x['is_healthy'])}")

    # 2) Load MD3VLM model + processor (once)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import md3vlm_baseline as genmod
    except Exception as e:
        print(f"ERROR importing md3vlm_baseline: {e}")
        raise

    print(f"\n📥 Loading MD3VLM from: {args.model_path}")
    model, processor, proj_out_num = genmod.load_md3vlm_model(args.model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"✅ Model loaded on device: {device}")

    # 3) Generation + evaluation loop
    print(f"\n🚀 Starting MD3VLM Evaluation on Test Set...")
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
            # Preprocess volume
            volume = genmod.load_and_preprocess_nifti(ex['image_path'], target_size=tuple(args.target_size))

            # Ensure shape (B, C, D, H, W)
            if volume.ndim == 3:
                volume = volume[np.newaxis, np.newaxis, ...]
            elif volume.ndim == 4:
                volume = volume[:, np.newaxis, ...]

            # Build prompt with image tokens
            image_tokens = "<im_patch>" * proj_out_num
            input_txt = image_tokens + genmod.DEFAULT_QUESTION

            input_ids = processor(input_txt, return_tensors="pt")["input_ids"].to(device)
            image_tensor = torch.from_numpy(volume).to(dtype=dtype, device=device)

            with torch.no_grad():
                generation = model.generate(
                    images=image_tensor,
                    inputs=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(not args.no_sampling),
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            report = processor.decode(generation[0], skip_special_tokens=True)

            gt_text = ex['report']
            gen_text = report

            all_refs.append(gt_text)
            all_hyps.append(gen_text)

            sample_res = {
                'patient_id': ex['patient_id'],
                'type': 'healthy' if ex['is_healthy'] else 'pathological',
                'gt': gt_text,
                'gen': gen_text,
            }

            c_scores = clin_metrics.compute_scores(gt_text, gen_text)
            sample_res.update(c_scores)
            for k, v in c_scores.items(): acc_clin[k].append(v)

            n_scores = nlg_calc.compute_sample(gt_text, gen_text)
            sample_res.update(n_scores)
            for k, v in n_scores.items(): acc_nlg[k].append(v)

            refs_cider[str(i)] = [gt_text]
            hyps_cider[str(i)] = [gen_text]

            results.append(sample_res)

        except Exception as e:
            print(f"Error {ex.get('patient_id','?')}: {e}")
            import traceback
            traceback.print_exc()

    # 4) BERTScore
    torch.cuda.empty_cache()
    bert_scores = compute_bertscore_safe(all_refs, all_hyps)
    if len(bert_scores) > 0:
        acc_nlg['bertscore'] = bert_scores

    # 5) Final statistics
    print("\n📊 Calculating Final Statistics...")
    final_stats = {
        'config': vars(args),
        'model_type': 'md3vlm_baseline',
        'metrics': {}
    }

    combined_acc = {**acc_clin, **acc_nlg}
    for k, values in combined_acc.items():
        mean = np.mean(values)
        low, high = bootstrap_ci(values)
        final_stats['metrics'][k] = {'mean': float(mean), 'ci_95': [float(low), float(high)]}

    if nlg_calc.cider is not None:
        cider = nlg_calc.compute_corpus_cider(refs_cider, hyps_cider)
        final_stats['metrics']['cider'] = {'mean': float(cider), 'ci_95': [0.0, 0.0]}

    # Save
    out_json = os.path.join(args.output_dir, 'metrics_md3vlm.json')
    with open(out_json, 'w') as f:
        json.dump(final_stats, f, indent=2)

    try:
        import pandas as _pd
        df = _pd.DataFrame(results)
        df.to_csv(os.path.join(args.output_dir, 'evaluation_samples_md3vlm.csv'), index=False)
    except Exception:
        pass

    print(f"💾 Results saved to: {args.output_dir}")

    # Print summary table
    print("\n" + "="*70)
    print("MD3VLM BASELINE - TEST SET EVALUATION")
    print("="*70)
    print(f"{'METRIC':<25} | {'MEAN':<10} | {'95% CI':<15}")
    print("-" * 70)
    metrics_order = [
        "bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "meteor", "bertscore", "cider",
        "clin_laterality_f1", "clin_anatomy_f1", "clin_pathology_f1"
    ]
    for m in metrics_order:
        if m in final_stats['metrics']:
            d = final_stats['metrics'][m]
            mean_val = d['mean']
            ci_low, ci_high = d['ci_95']
            print(f"{m.upper():<25} | {mean_val:.4f}     | [{ci_low:.3f}, {ci_high:.3f}]")
    print("="*70)

if __name__ == '__main__':
    main()
