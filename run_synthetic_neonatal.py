#!/usr/bin/env python
"""
Simple script to train CT model on synthetic neonatal data
"""
import sys
import subprocess
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from datetime import datetime
import os
import yaml
from pathlib import Path

def extract_metrics_from_output(output):
    """Extract test metrics from training output"""
    metrics = {}
    
    lines = output.split('\n')
    
    # Extract test metrics only (no validation)
    for i, line in enumerate(lines):
        # Extract AUC metrics for test set
        if 'Test' in line and 'AUC-ROC:' in line and 'AUC-PR:' in line:
            match = re.search(r'Test ([\w_]+) AUC-ROC: ([\d.]+); AUC-PR: ([\d.]+)', line)
            if match:
                outcome_name = match.group(1)
                metrics[f'test_{outcome_name}_auc_roc'] = float(match.group(2))
                metrics[f'test_{outcome_name}_auc_pr'] = float(match.group(3))
        
        # Extract RMSE metrics
        if 'Test' in line and 'normalised RMSE' in line:
            match = re.search(r'Test ([\w_]+) normalised RMSE \(all\): ([\d.]+); \(orig\): ([\d.]+)', line)
            if match:
                outcome_name = match.group(1)
                metrics[f'test_{outcome_name}_rmse_all'] = float(match.group(2))
                metrics[f'test_{outcome_name}_rmse_orig'] = float(match.group(3))
        
        # Extract from test results dictionary
        if "'multi_test_" in line:
            # Extract all metrics from dictionary output
            # Look for pattern 'key': value
            for j in range(max(0, i-5), min(len(lines), i+10)):
                dict_line = lines[j]
                matches = re.findall(r"'([\w_]+)': ([\d.-]+)", dict_line)
                for key, value in matches:
                    if 'multi_test_' in key:
                        metrics[key] = float(value)
    
    return metrics

def extract_autoregressive_metrics(output):
    """Extract metrics from simple_autoregressive_predict.py output"""
    metrics = {}
    
    lines = output.split('\n')
    
    # Extract metrics for each day before diagnosis
    # Pattern: "N days before end (n=X patients - Y cases, Z controls):"
    #          "  Disease AUC-ROC: 0.XXXX"
    #          "  Disease AUC-PR: 0.XXXX"
    
    for i, line in enumerate(lines):
        match = re.search(r'(\d+) days before end.*?(\d+\.?\d*) cases.*?(\d+\.?\d*) controls', line)
        if match:
            days_before = int(match.group(1))
            
            # Look for AUC metrics in the next few lines
            for j in range(i+1, min(i+5, len(lines))):
                auc_line = lines[j]
                
                # Extract AUC-ROC (match any word before AUC-ROC)
                auc_roc_match = re.search(r'[\w_]+ AUC-ROC: ([\d.]+)', auc_line)
                if auc_roc_match:
                    metrics[f'day_{days_before}_auc_roc'] = float(auc_roc_match.group(1))
                
                # Extract AUC-PR (match any word before AUC-PR)
                auc_pr_match = re.search(r'[\w_]+ AUC-PR: ([\d.]+)', auc_line)
                if auc_pr_match:
                    metrics[f'day_{days_before}_auc_pr'] = float(auc_pr_match.group(1))
                
                # Extract Prevalence
                prevalence_match = re.search(r'Prevalence: ([\d.]+)', auc_line)
                if prevalence_match:
                    metrics[f'day_{days_before}_prevalence'] = float(prevalence_match.group(1))
    
    return metrics

def main(repeats=1, max_epochs=1):
    """Run training multiple times and report mean/std of metrics"""
    
    # Load the config to get the target outcome
    config_path = Path(__file__).parent / 'config' / 'dataset' / 'synthetic_neonatal.yaml'
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Extract the disease name from config
    target_outcome = dataset_config.get('dataset', {}).get('target_disease', 'dated_max_chole_TPNEHR')
    # Remove 'dated_' prefix to get disease name
    disease = target_outcome.replace('dated_', '') if target_outcome.startswith('dated_') else target_outcome
    
    # Check if mute_decoder is enabled
    mute_decoder = dataset_config.get('dataset', {}).get('mute_decoder', False)
    if mute_decoder:
        print(f"Note: mute_decoder is enabled - autoregressive outputs will be zeroed during training")
    
    all_metrics = defaultdict(list)
    
    for repeat in range(repeats):
        print(f"\n{'='*60}")
        print(f"REPEAT {repeat + 1}/{repeats}")
        print(f"{'='*60}\n")
        
        # Use different seed for each repeat
        seed = 42 + repeat
        
        cmd = [
            sys.executable, "-m", "runnables.train_multi",
            "+backbone=ct",
            "+dataset=synthetic_neonatal",
            "+backbone/ct_hparams=synthetic_neonatal",
            "exp.gpus=null",  # Use CPU
            "exp.logging=False",  # Disable MLflow logging
            f"exp.seed={seed}",
            f"exp.max_epochs={max_epochs}"
        ]
        
        print("Running command:", " ".join(cmd))
        
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        # Check return code
        if result.returncode != 0:
            print(f"ERROR: Training script failed with return code {result.returncode}")
            print(f"STDERR:\n{result.stderr}")
            print(f"STDOUT:\n{result.stdout}")
            # Continue to next repeat instead of stopping
            continue
        
        # Debug: Print first 1000 chars of output to see what's happening
        if len(output.strip()) == 0:
            print("WARNING: No output from training script!")
        else:
            print(f"Output preview (first 1000 chars):\n{output[:1000]}...")
        
        # Extract metrics from output
        metrics = extract_metrics_from_output(output)
        
        # Store metrics
        for key, value in metrics.items():
            all_metrics[key].append(value)
        
        print(f"\nRepeat {repeat + 1} training metrics extracted: {len(metrics)} metrics")
        
        # Show key metrics for this run
        if f'test_{disease}_auc_roc' in metrics:
            print(f"  Test AUC-ROC: {metrics[f'test_{disease}_auc_roc']:.4f}")
            print(f"  Test AUC-PR: {metrics[f'test_{disease}_auc_pr']:.4f}")
        if 'multi_test_loss' in metrics:
            print(f"  Test Loss: {metrics['multi_test_loss']:.4f}")
        
        # Run simple_autoregressive_predict.py
        print(f"\nRunning simple_autoregressive_predict.py for repeat {repeat + 1}...")
        predict_cmd = [sys.executable, "simple_autoregressive_predict.py"]
        predict_result = subprocess.run(predict_cmd, capture_output=True, text=True)
        predict_output = predict_result.stdout + predict_result.stderr
        
        # Debug: Print autoregressive output
        if len(predict_output.strip()) == 0:
            print("WARNING: No output from autoregressive script!")
        elif "Error" in predict_output or "error" in predict_output:
            print(f"ERROR in autoregressive script:\n{predict_output[:1000]}...")
        
        # Extract autoregressive metrics
        auto_metrics = extract_autoregressive_metrics(predict_output)
        
        # Debug: print extracted metrics
        print(f"DEBUG: Extracted autoregressive metrics: {sorted(auto_metrics.keys())}")
        
        # Store autoregressive metrics
        for key, value in auto_metrics.items():
            all_metrics[key].append(value)
        
        print(f"Autoregressive metrics extracted: {len(auto_metrics)} metrics")
        
        # Show key autoregressive metrics
        if 'day_0_auc_roc' in auto_metrics:
            print(f"  Day 0 AUC-ROC: {auto_metrics['day_0_auc_roc']:.4f}")
            print(f"  Day 0 AUC-PR: {auto_metrics['day_0_auc_pr']:.4f}")
    
    # Print summary with lists of values
    print(f"\n{'='*60}")
    print(f"SUMMARY OVER {repeats} REPEATS")
    print(f"{'='*60}\n")
    
    # Group metrics by category
    training_metrics = {}
    autoregressive_metrics = {}
    
    for metric_name, values in all_metrics.items():
        if 'day_' in metric_name:
            autoregressive_metrics[metric_name] = values
        else:
            training_metrics[metric_name] = values
    
    # Print training metrics
    if training_metrics:
        print("TRAINING METRICS:")
        for metric_name in sorted(training_metrics.keys()):
            values = training_metrics[metric_name]
            if len(values) == repeats:
                values_str = ', '.join([f"{v:.4f}" for v in values])
                print(f"  {metric_name}: [{values_str}]")
            else:
                print(f"  {metric_name}: incomplete data ({len(values)}/{repeats} runs)")
    
    # Print autoregressive metrics by day
    if autoregressive_metrics:
        print("\nAUTOREGRESSIVE PREDICTION METRICS:")
        for day in range(0, 16, 2):  # 0-14 days, step by 2
            print(f"\n  {day} days before diagnosis:")
            
            # AUC-ROC
            roc_key = f'day_{day}_auc_roc'
            if roc_key in autoregressive_metrics:
                values = autoregressive_metrics[roc_key]
                if len(values) == repeats:
                    values_str = ', '.join([f"{v:.4f}" for v in values])
                    print(f"    AUC-ROC: [{values_str}]")
            
            # AUC-PR
            pr_key = f'day_{day}_auc_pr'
            if pr_key in autoregressive_metrics:
                values = autoregressive_metrics[pr_key]
                if len(values) == repeats:
                    values_str = ', '.join([f"{v:.4f}" for v in values])
                    print(f"    AUC-PR:  [{values_str}]")
            
            # Prevalence
            prev_key = f'day_{day}_prevalence'
            if prev_key in autoregressive_metrics:
                values = autoregressive_metrics[prev_key]
                if len(values) == repeats:
                    values_str = ', '.join([f"{v:.4f}" for v in values])
                    print(f"    Prevalence: [{values_str}]")
    
    # Save results to DataFrame
    save_results_to_dataframe(all_metrics, repeats, max_epochs, disease)

def save_results_to_dataframe(all_metrics, repeats, max_epochs, outcome_name):
    """Save results to a structured DataFrame"""
    
    # Debug: print what metrics we have
    print(f"\nDEBUG: all_metrics keys: {sorted([k for k in all_metrics.keys() if 'day_' in k])}")
    
    # Create summary data
    rows = []
    for repeat_idx in range(repeats):
        row = {
            'repeat': repeat_idx + 1,
            'seed': 42 + repeat_idx,
        }
        
        # Add autoregressive metrics for each day
        for day in range(0, 16, 2):  # 0-14 days, step by 2
            roc_key = f'day_{day}_auc_roc'
            pr_key = f'day_{day}_auc_pr'
            prev_key = f'day_{day}_prevalence'
            
            # Initialize with None to ensure all columns exist
            row[f'auc_roc_day{day}'] = None
            row[f'auc_pr_day{day}'] = None
            row[f'prevalence_day{day}'] = None
            
            if roc_key in all_metrics and len(all_metrics[roc_key]) > repeat_idx:
                row[f'auc_roc_day{day}'] = all_metrics[roc_key][repeat_idx]
            
            if pr_key in all_metrics and len(all_metrics[pr_key]) > repeat_idx:
                row[f'auc_pr_day{day}'] = all_metrics[pr_key][repeat_idx]
            
            if prev_key in all_metrics and len(all_metrics[prev_key]) > repeat_idx:
                row[f'prevalence_day{day}'] = all_metrics[prev_key][repeat_idx]
        
        rows.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(rows)
    filename = f"outputs/{outcome_name}.csv"
    summary_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train CT model on synthetic neonatal data')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat training (default: 1)')
    parser.add_argument('--max-epochs', type=int, default=1000, help='Maximum number of epochs per training run (default: 1)')
    args = parser.parse_args()
    
    main(repeats=args.repeats, max_epochs=args.max_epochs)