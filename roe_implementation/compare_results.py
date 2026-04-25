"""
Compare baseline vs RoE generation results.

This script helps you evaluate the impact of RoE on your Open-RAG model.
"""

import json
import argparse
from typing import Dict, List
import numpy as np


def load_results(filepath: str) -> Dict:
    """Load results from JSONL or JSON file"""
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        return data[0] if len(data) == 1 else data
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def compare_metrics(baseline: Dict, roe: Dict):
    """Compare metrics between baseline and RoE"""
    
    print("="*80)
    print("METRIC COMPARISON")
    print("="*80)
    
    # Overall metrics
    if 'metric_mean' in baseline and 'metric_mean' in roe:
        baseline_score = baseline['metric_mean']
        roe_score = roe['metric_mean']
        improvement = ((roe_score - baseline_score) / baseline_score) * 100
        
        print(f"\nOverall Score:")
        print(f"  Baseline: {baseline_score:.4f}")
        print(f"  RoE:      {roe_score:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    # F1, EM, Precision, Recall for HotpotQA
    if 'F1' in baseline and 'F1' in roe:
        metrics = ['F1', 'EM', 'Precision', 'Recall']
        print(f"\nDetailed Metrics:")
        
        for metric in metrics:
            if metric in baseline and metric in roe:
                baseline_vals = baseline[metric]
                roe_vals = roe[metric]
                
                baseline_mean = np.mean(baseline_vals)
                roe_mean = np.mean(roe_vals)
                improvement = ((roe_mean - baseline_mean) / baseline_mean) * 100
                
                print(f"\n  {metric}:")
                print(f"    Baseline: {baseline_mean:.4f}")
                print(f"    RoE:      {roe_mean:.4f}")
                print(f"    Improvement: {improvement:+.2f}%")


def compare_predictions(baseline: Dict, roe: Dict, num_examples: int = 10):
    """Compare individual predictions"""
    
    print("\n" + "="*80)
    print("PREDICTION COMPARISON (First {} examples)".format(num_examples))
    print("="*80)
    
    baseline_preds = baseline.get('preds', [])
    roe_preds = roe.get('preds', [])
    prompts = baseline.get('prompts', [])
    
    for i in range(min(num_examples, len(baseline_preds), len(roe_preds))):
        print(f"\n--- Example {i+1} ---")
        
        if prompts and i < len(prompts):
            # Extract question from prompt
            prompt = prompts[i]
            if "### Instruction:" in prompt:
                question = prompt.split("### Instruction:")[1].split("### Response:")[0].strip()
                print(f"Question: {question[:200]}...")
        
        print(f"Baseline: {baseline_preds[i]}")
        print(f"RoE:      {roe_preds[i]}")
        
        # Check if they differ
        if baseline_preds[i] != roe_preds[i]:
            print("  ⚠️  Different predictions!")
        else:
            print("  ✓ Same prediction")


def analyze_disagreements(baseline: Dict, roe: Dict):
    """Analyze cases where RoE and baseline disagree"""
    
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS")
    print("="*80)
    
    baseline_preds = baseline.get('preds', [])
    roe_preds = roe.get('preds', [])
    baseline_correct = baseline.get('metric_results', [])
    roe_correct = roe.get('metric_results', [])
    
    if not (baseline_preds and roe_preds and baseline_correct and roe_correct):
        print("Insufficient data for disagreement analysis")
        return
    
    # Categories
    both_correct = 0
    both_wrong = 0
    baseline_only = 0
    roe_only = 0
    
    roe_fixes = []  # Cases where RoE fixes baseline errors
    roe_breaks = []  # Cases where RoE breaks baseline correct answers
    
    for i in range(min(len(baseline_preds), len(roe_preds))):
        b_pred = baseline_preds[i]
        r_pred = roe_preds[i]
        b_corr = baseline_correct[i] if i < len(baseline_correct) else 0
        r_corr = roe_correct[i] if i < len(roe_correct) else 0
        
        if b_corr and r_corr:
            both_correct += 1
        elif not b_corr and not r_corr:
            both_wrong += 1
        elif b_corr and not r_corr:
            baseline_only += 1
            roe_breaks.append(i)
        elif not b_corr and r_corr:
            roe_only += 1
            roe_fixes.append(i)
    
    total = both_correct + both_wrong + baseline_only + roe_only
    
    print(f"\nTotal examples: {total}")
    print(f"\nBreakdown:")
    print(f"  Both correct:     {both_correct:4d} ({both_correct/total*100:.1f}%)")
    print(f"  Both wrong:       {both_wrong:4d} ({both_wrong/total*100:.1f}%)")
    print(f"  Baseline only:    {baseline_only:4d} ({baseline_only/total*100:.1f}%)")
    print(f"  RoE only:         {roe_only:4d} ({roe_only/total*100:.1f}%)")
    
    print(f"\nNet improvement: {roe_only - baseline_only:+d} examples")
    
    if roe_fixes:
        print(f"\n✅ RoE fixed {len(roe_fixes)} baseline errors:")
        print(f"   Example indices: {roe_fixes[:10]}...")
    
    if roe_breaks:
        print(f"\n❌ RoE broke {len(roe_breaks)} baseline correct answers:")
        print(f"   Example indices: {roe_breaks[:10]}...")
    
    # Show examples of fixes
    if roe_fixes:
        print(f"\n{'='*80}")
        print(f"EXAMPLES OF RoE FIXES (First 3)")
        print(f"{'='*80}")
        
        prompts = baseline.get('prompts', [])
        
        for i, idx in enumerate(roe_fixes[:3]):
            print(f"\n--- Fix Example {i+1} (Index {idx}) ---")
            
            if prompts and idx < len(prompts):
                prompt = prompts[idx]
                if "### Instruction:" in prompt:
                    question = prompt.split("### Instruction:")[1].split("### Response:")[0].strip()
                    print(f"Question: {question[:200]}...")
            
            print(f"Baseline (Wrong): {baseline_preds[idx]}")
            print(f"RoE (Correct):    {roe_preds[idx]}")


def analyze_retrieval_decisions(baseline: Dict, roe: Dict):
    """Analyze retrieval decision differences"""
    
    print("\n" + "="*80)
    print("RETRIEVAL DECISION ANALYSIS")
    print("="*80)
    
    # Check if retrieval decision data is available
    baseline_results = baseline.get('all_results', [])
    roe_results = roe.get('all_results', [])
    
    if not baseline_results or not roe_results:
        print("No retrieval decision data available")
        return
    
    baseline_retrieve = 0
    roe_retrieve = 0
    
    for i in range(min(len(baseline_results), len(roe_results))):
        b_res = baseline_results[i]
        r_res = roe_results[i]
        
        # Check if retrieval was used
        if isinstance(b_res, dict):
            if any(k.startswith('retrieval_') for k in b_res.keys()):
                baseline_retrieve += 1
        
        if isinstance(r_res, dict):
            if any(k.startswith('retrieval_') for k in r_res.keys()):
                roe_retrieve += 1
    
    total = len(baseline_results)
    
    print(f"\nRetrieval frequency:")
    print(f"  Baseline: {baseline_retrieve}/{total} ({baseline_retrieve/total*100:.1f}%)")
    print(f"  RoE:      {roe_retrieve}/{total} ({roe_retrieve/total*100:.1f}%)")
    print(f"  Difference: {roe_retrieve - baseline_retrieve:+d} ({(roe_retrieve-baseline_retrieve)/total*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs RoE results")
    parser.add_argument("baseline_file", type=str, help="Baseline results file")
    parser.add_argument("roe_file", type=str, help="RoE results file")
    parser.add_argument("--num_examples", type=int, default=10, 
                       help="Number of examples to show")
    
    args = parser.parse_args()
    
    print("Loading results...")
    baseline = load_results(args.baseline_file)
    roe = load_results(args.roe_file)
    
    print(f"Baseline file: {args.baseline_file}")
    print(f"RoE file:      {args.roe_file}")
    
    # Compare metrics
    compare_metrics(baseline, roe)
    
    # Compare predictions
    compare_predictions(baseline, roe, args.num_examples)
    
    # Analyze disagreements
    analyze_disagreements(baseline, roe)
    
    # Analyze retrieval decisions
    analyze_retrieval_decisions(baseline, roe)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
