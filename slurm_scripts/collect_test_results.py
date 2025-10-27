#!/usr/bin/env python3
"""
Script to collect test evaluation results from the testing directory.
Extracts F1 scores and other metrics from metrics.json files for each model/subset combination.
"""

import os
import json
import pandas as pd
from pathlib import Path

# Configuration
TESTING_DIR = "/net/tscratch/people/plgjmachali/surgvu_results/testing/job_1955382_20251019_1506"


def find_metrics_files(base_dir):
    """Find all metrics.json files in the testing directory structure."""
    metrics_files = []
    
    print(f"🔍 Scanning directory: {base_dir}")
    
    # Expected structure: base_dir/model/subset/metrics.json
    if not os.path.exists(base_dir):
        print(f"❌ Directory does not exist: {base_dir}")
        return metrics_files
    
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        print(f"📁 Found model: {model_dir}")
        
        for subset_dir in os.listdir(model_path):
            subset_path = os.path.join(model_path, subset_dir)
            if not os.path.isdir(subset_path):
                continue
                
            print(f"  📂 Found subset: {subset_dir}")
            
            # Look for metrics.json in this directory
            metrics_file = os.path.join(subset_path, "metrics.json")
            if os.path.exists(metrics_file):
                metrics_files.append({
                    "model": model_dir,
                    "subset": subset_dir,
                    "metrics_file": metrics_file,
                    "result_dir": subset_path
                })
                print(f"    ✅ Found metrics.json")
            else:
                print(f"    ❌ No metrics.json found")
    
    print(f"\n📊 Found {len(metrics_files)} metrics files total")
    return metrics_files

def extract_test_metrics(metrics_file):
    """Extract test metrics from metrics.json file."""
    print(f"\n🔬 Processing: {metrics_file}")
    
    try:
        with open(metrics_file, "r") as f:
            content = f.read().strip()
        
        # Handle multi-line JSON (each line is a separate JSON object)
        metrics_list = []
        if '\n' in content:
            print("  Multi-line JSON detected")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    try:
                        line_data = json.loads(line.strip())
                        metrics_list.append(line_data)
                    except json.JSONDecodeError as line_error:
                        print(f"    ⚠️  Error parsing line {i+1}: {line_error}")
                        continue
        else:
            print("  Single JSON object detected")
            metrics_list = [json.loads(content)]
        
        print(f"  Loaded {len(metrics_list)} metric entries")
        
        # Look for test entries (they should contain test_f1_score_list_meter)
        test_entries = [entry for entry in metrics_list if "test_f1_score_list_meter" in entry]
        print(f"  Found {len(test_entries)} test entries")
        
        if not test_entries:
            print("  ❌ No test entries found")
            return None
        
        # For test evaluation, we typically expect only one test entry
        # Take the first (or only) test entry
        test_entry = test_entries[0]
        
        def extract_metric_value(data, meter_name, metric_type):
            """Helper function to extract metric values"""
            if meter_name not in data:
                print(f"    ⚠️  Warning: Meter '{meter_name}' not found in data")
                print(f"        Available meters: {list(data.keys())}")
                return -1
            meter_data = data[meter_name]
            if not isinstance(meter_data, dict):
                print(f"    ⚠️  Warning: Meter '{meter_name}' data is not a dict: {type(meter_data)}")
                return -1
            if metric_type not in meter_data:
                print(f"    ⚠️  Warning: Metric type '{metric_type}' not found in meter '{meter_name}'")
                print(f"        Available metric types: {list(meter_data.keys())}")
                return -1
            
            metric_value = meter_data[metric_type]
            # Handle different formats
            if isinstance(metric_value, dict) and "0" in metric_value:
                return metric_value["0"]
            elif isinstance(metric_value, (int, float)):
                return metric_value
            else:
                print(f"    ⚠️  Warning: Unknown metric value format for '{meter_name}.{metric_type}': {type(metric_value)}")
                return -1
        
        # Extract metrics
        test_f1_macro = extract_metric_value(test_entry, "test_f1_score_list_meter", "macro")
        test_f1_weighted = extract_metric_value(test_entry, "test_f1_score_list_meter", "weighted")
        test_accuracy = extract_metric_value(test_entry, "test_accuracy_list_meter", "top_1")
        
        # Get phase information
        phase_idx = test_entry.get("phase_idx", test_entry.get("train_phase_idx", 0))
        
        print(f"  ✅ Extracted metrics:")
        print(f"    Phase: {phase_idx}")
        print(f"    F1 Macro: {test_f1_macro:.6f}")
        print(f"    F1 Weighted: {test_f1_weighted:.6f}")
        print(f"    Accuracy: {test_accuracy:.6f}")
        
        return {
            "phase": phase_idx,
            "test_f1_macro": test_f1_macro,
            "test_f1_weighted": test_f1_weighted,
            "test_accuracy": test_accuracy,
            "raw_f1_data": test_entry.get("test_f1_score_list_meter", {}),
            "raw_accuracy_data": test_entry.get("test_accuracy_list_meter", {})
        }
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ Error reading {metrics_file}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error processing {metrics_file}: {e}")
        return None

def main():
    """Main function to collect and analyze test results."""
    print("🚀 Collecting test evaluation results...")
    print(f"📁 Testing directory: {TESTING_DIR}")
    
    # Find all metrics files
    metrics_files = find_metrics_files(TESTING_DIR)
    
    if not metrics_files:
        print("❌ No metrics files found!")
        return
    
    # Process each metrics file
    results = []
    successful_extractions = 0
    
    for file_info in metrics_files:
        model = file_info["model"]
        subset = file_info["subset"]
        metrics_file = file_info["metrics_file"]
        result_dir = file_info["result_dir"]
        
        print(f"\n" + "="*60)
        print(f"Processing: {model} / {subset}")
        
        # Extract metrics
        metrics = extract_test_metrics(metrics_file)
        
        if metrics:
            # Map subset numbers to percentages for display
            subset_display = {
                "12": "12.5%",
                "25": "25%", 
                "100": "100%"
            }.get(subset, f"{subset}%")
            
            # Map model names for display
            model_display = {
                "imagenet_to_surgvu": "ImageNet",
                "moco_to_surgvu": "MoCo",
                "simclr_to_surgvu": "SimCLR"
            }.get(model, model)
            
            results.append({
                "model": model_display,
                "model_original": model,
                "data_subset": subset_display,
                "subset_original": subset,
                "phase": metrics["phase"],
                "test_f1_macro": metrics["test_f1_macro"],
                "test_f1_weighted": metrics["test_f1_weighted"],
                "test_accuracy": metrics["test_accuracy"],
                "result_directory": result_dir,
                "metrics_file": metrics_file
            })
            
            successful_extractions += 1
            print(f"  ✅ Successfully processed {model}/{subset}")
        else:
            print(f"  ❌ Failed to process {model}/{subset}")
    
    # Create DataFrame and save results
    if results:
        df = pd.DataFrame(results)
        
        # Sort by model and subset for better readability
        model_order = ["ImageNet", "MoCo", "SimCLR"]
        subset_order = ["12.5%", "25%", "100%"]
        
        df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
        df["data_subset"] = pd.Categorical(df["data_subset"], categories=subset_order, ordered=True)
        df = df.sort_values(["model", "data_subset"]).reset_index(drop=True)
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"="*60)
        print(f"Successfully processed: {successful_extractions}/{len(metrics_files)} combinations")
        print()
        
        # Print results table
        display_columns = ["model", "data_subset", "test_f1_macro", "test_f1_weighted", "test_accuracy"]
        print("Results:")
        print(df[display_columns].to_string(index=False, float_format="%.4f"))
        
        # Save to CSV
        output_csv = "test_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")
        
        # Additional summary statistics
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        for model in model_order:
            model_data = df[df["model"] == model]
            if len(model_data) > 0:
                avg_f1 = model_data["test_f1_macro"].mean()
                max_f1 = model_data["test_f1_macro"].max()
                min_f1 = model_data["test_f1_macro"].min()
                print(f"{model:>8}: Avg F1={avg_f1:.4f}, Max={max_f1:.4f}, Min={min_f1:.4f}")
        
        print(f"\n🏆 Best overall F1 macro: {df['test_f1_macro'].max():.4f}")
        best_row = df.loc[df['test_f1_macro'].idxmax()]
        print(f"   Model: {best_row['model']}, Subset: {best_row['data_subset']}")
        
    else:
        print("\n❌ No results to save!")
    
    print(f"\n✨ Analysis completed!")

if __name__ == "__main__":
    main()