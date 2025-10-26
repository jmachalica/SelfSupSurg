import os
import json
import pandas as pd

# Define the base directory for results
RESULTS_DIR = "/net/tscratch/people/plgjmachali/surgvu_results/finetuning/"

# Define the models and data subsets to analyze
MODELS = ["imagenet_to_surgvu", "moco_to_surgvu", "simclr_to_surgvu"]
DATA_SUBSETS = ["12", "25", "100"]

# Initialize an empty list to store results
results = []

# Iterate over models and data subsets
for model in MODELS:
    for subset in DATA_SUBSETS:
        subset_dir = os.path.join(RESULTS_DIR, model, subset)
        print(f"\n🔍 Checking: {model}/{subset}")
        print(f"  Directory: {subset_dir}")
        if not os.path.exists(subset_dir):
            print(f"  ❌ WARNING: Directory does not exist - SKIPPING")
            continue

        # Find the newest job directory
        job_dirs = [
            os.path.join(subset_dir, d) for d in os.listdir(subset_dir)
            if os.path.isdir(os.path.join(subset_dir, d))
        ]
        print(f"Found {len(job_dirs)} job directories")
        if not job_dirs:
            print(f"  ❌ WARNING: No job directories found - SKIPPING")
            continue

        newest_job_dir = max(job_dirs, key=os.path.getmtime)
        metrics_file = os.path.join(newest_job_dir, "metrics.json")
        job_name = os.path.basename(newest_job_dir)
        print(f"  Newest job: {job_name}")
        print(f"  Metrics file: {metrics_file}")

        # Check if metrics.json exists
        if not os.path.exists(metrics_file):
            print(f"  ❌ WARNING: metrics.json does not exist - SKIPPING")
            continue

        # Load metrics.json and extract the best F1 macro score
        try:
            with open(metrics_file, "r") as f:
                content = f.read().strip()
                
            # Handle multi-line JSON (each line is a separate JSON object representing a phase/iteration)
            metrics_list = []
            if '\n' in content:
                print(f"Multi-line JSON detected, parsing line by line...")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        try:
                            line_data = json.loads(line.strip())
                            metrics_list.append(line_data)
                        except json.JSONDecodeError as line_error:
                            print(f"  ⚠️  WARNING: Error parsing line {i+1}: {line_error}")
                            print(f"  Line content: {line[:100]}...")
                            continue
            else:
                print(f"Single JSON object detected")
                # Single JSON object
                metrics_list = [json.loads(content)]
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"❌ WARNING: Error reading {metrics_file}: {e}")
            print(f"File exists: {os.path.exists(metrics_file)}")
            if os.path.exists(metrics_file):
                print(f"File size: {os.path.getsize(metrics_file)} bytes")
                # Show first few lines of the file
                try:
                    with open(metrics_file, "r") as f:
                        first_lines = f.read(500)
                        print(f"First 500 chars: {first_lines}")
                except Exception as read_error:
                    print(f"WARNING: Could not read file content: {read_error}")
            print(f"❌ WARNING: SKIPPING {model}/{subset} due to JSON error")
            continue

        # Find the phase with the best test F1 macro score
        best_f1_macro = -1
        best_phase = None
        best_train_phase = None
        best_data = None
        
        print(f"\nProcessing: {model}/{subset}")
        print(f"Metrics file: {metrics_file}")
        print(f"Loaded {len(metrics_list)} metric entries")
        
        # Look for test entries (they contain test_f1_score_list_meter)
        test_entries = [entry for entry in metrics_list if "test_f1_score_list_meter" in entry]
        print(f"Found {len(test_entries)} test entries")
        
        for entry in test_entries:
            if "test_f1_score_list_meter" in entry:
                f1_data = entry["test_f1_score_list_meter"]
                
                # Validate required indices
                if "phase_idx" not in entry:
                    print(f"  ❌ WARNING: Missing phase_idx in test entry: {entry}")
                    continue
                if "train_phase_idx" not in entry:
                    print(f"  ❌ WARNING: Missing train_phase_idx in test entry: {entry}")
                    continue
                
                phase_idx = entry["phase_idx"]
                train_phase_idx = entry["train_phase_idx"]
                
                # print(f"  Test entry for phase {phase_idx} (train_phase {train_phase_idx})")
                # print(f"    F1 data: {f1_data}")
                
                if isinstance(f1_data, dict) and "macro" in f1_data:
                    macro_data = f1_data["macro"]
                    # print(f"    Macro data: {macro_data} (type: {type(macro_data)})")
                    
                    # Handle different formats: dict with "0" key or direct value
                    if isinstance(macro_data, dict) and "0" in macro_data:
                        f1_macro = macro_data["0"]
                    elif isinstance(macro_data, (int, float)):
                        f1_macro = macro_data
                    else:
                        print(f"    ❌ WARNING: Unknown macro format: {type(macro_data)}")
                        continue
                    
                    # print(f"    Extracted F1 macro: {f1_macro}")
                    
                    if f1_macro > best_f1_macro:
                        best_f1_macro = f1_macro
                        best_phase = phase_idx
                        best_train_phase = train_phase_idx
                        best_data = entry
                        # print(f"  ✅ New best: Phase {phase_idx} (train_phase {train_phase_idx}), F1 Macro: {f1_macro}")
                else:
                    print(f"    ❌ WARNING: test_f1_score_list_meter missing 'macro' key or invalid format")

        print(f"🏆 Final best: Phase {best_phase} (train_phase {best_train_phase}), F1 Macro: {best_f1_macro}")

        # Validate that we found a valid best phase
        if best_train_phase is None or best_phase is None:
            print(f"❌ WARNING: No valid test entries found with required indices for {model}/{subset}")
            print(f"  Skipping this model/subset combination")
            continue

        # Construct checkpoint path based on best train_phase
        checkpoint_file = None
        # Common checkpoint naming patterns (based on train_phase_idx)
        possible_checkpoint_names = [
            f"model_final_checkpoint_phase{best_train_phase}.torch",
            f"model_phase{best_train_phase}.torch",
            f"checkpoint_phase{best_train_phase}.torch",
            f"model_final_checkpoint_phase{best_train_phase}.pth",
            f"model_phase{best_train_phase}.pth",
            f"checkpoint_phase{best_train_phase}.pth"
        ]
        
        for checkpoint_name in possible_checkpoint_names:
            potential_path = os.path.join(newest_job_dir, checkpoint_name)
            if os.path.exists(potential_path):
                checkpoint_file = potential_path
                print(f"✅ Found checkpoint: {checkpoint_name}")
                break
        
        if checkpoint_file is None:
            print(f"⚠️  WARNING: No checkpoint file found for train_phase {best_train_phase}")
            print(f"    Searched in: {newest_job_dir}")
            # List available checkpoint files for debugging
            checkpoint_files = [f for f in os.listdir(newest_job_dir) 
                                if f.endswith(('.torch', '.pth')) and 'checkpoint' in f.lower()]
            if checkpoint_files:
                print(f"    Available checkpoint files: {checkpoint_files}")
            else:
                print(f"    WARNING: No checkpoint files found in directory")        # Extract all metrics from the best phase

        if best_data is not None:
            print(f"Extracting metrics from best phase {best_phase} (train_phase {best_train_phase})")
            
            def extract_metric_value(data, meter_name, metric_type):
                """Helper function to extract metric values"""
                if meter_name not in data:
                    return -1
                meter_data = data[meter_name]
                if not isinstance(meter_data, dict) or metric_type not in meter_data:
                    return -1
                
                metric_value = meter_data[metric_type]
                # Handle different formats
                if isinstance(metric_value, dict) and "0" in metric_value:
                    return metric_value["0"]
                elif isinstance(metric_value, (int, float)):
                    return metric_value
                else:
                    return -1
            
            # Test metrics from best phase
            test_f1_weighted = extract_metric_value(best_data, "test_f1_score_list_meter", "weighted")
            test_accuracy = extract_metric_value(best_data, "test_accuracy_list_meter", "top_1")
            
            # Find corresponding train metrics for the same train_phase
            # Only consider entries that have the required train_phase_idx
            train_entries = [entry for entry in metrics_list 
                           if "train_f1_score_list_meter" in entry and 
                           "train_phase_idx" in entry and
                           entry["train_phase_idx"] == best_train_phase]
            
            if train_entries:
                train_data = train_entries[0]  # Take the first matching train entry
                train_f1_macro = extract_metric_value(train_data, "train_f1_score_list_meter", "macro")
                train_f1_weighted = extract_metric_value(train_data, "train_f1_score_list_meter", "weighted")
                train_accuracy = extract_metric_value(train_data, "train_accuracy_list_meter", "top_1")
            else:
                train_f1_macro = train_f1_weighted = train_accuracy = -1
                print(f"  ⚠️  WARNING: No train data found for train_phase {best_train_phase}")
            
            print(f"  Test metrics - F1 W: {test_f1_weighted}, Acc: {test_accuracy}")
            print(f"  Train metrics - F1 M: {train_f1_macro}, F1 W: {train_f1_weighted}, Acc: {train_accuracy}")
        else:
            test_f1_weighted = test_accuracy = train_f1_macro = train_f1_weighted = train_accuracy = -1
            print(f"  WARNING: No valid test data found!")

        # Append the result to the list
        results.append({
            "model": model,
            "data_subset": subset,
            "job_directory": job_name,
            "job_path": newest_job_dir,
            "checkpoint_path": checkpoint_file,
            "checkpoint_exists": checkpoint_file is not None and os.path.exists(checkpoint_file) if checkpoint_file else False,
            "best_phase": best_phase,
            "best_train_phase": best_train_phase,
            "test_f1_macro": best_f1_macro,
            "test_f1_weighted": test_f1_weighted,
            "test_accuracy": test_accuracy,
            "train_f1_macro": train_f1_macro,
            "train_f1_weighted": train_f1_weighted,
            "train_accuracy": train_accuracy
        })
        print(f"✅ Successfully processed {model}/{subset}")

print(f"\n📊 SUMMARY: Processed {len(results)} model/subset combinations")
print("Results breakdown:")
for result in results:
    print(f"  - {result['model']}/{result['data_subset']}: F1 macro = {result['test_f1_macro']:.4f}")

print(f"\n🗂️  CHECKPOINT SUMMARY:")
print("Selected checkpoints for each model/subset:")
for result in results:
    print(f"  📁 {result['model']}/{result['data_subset']}:")
    print(f"    Job Directory: {result['job_directory']}")
    print(f"    Job Path: {result['job_path']}")
    print(f"    Best Phase: {result['best_phase']} (train_phase: {result['best_train_phase']})")
    if result['checkpoint_path']:
        print(f"    Checkpoint: {os.path.basename(result['checkpoint_path'])}")
        print(f"    Full Checkpoint Path: {result['checkpoint_path']}")
        print(f"    Exists: {'✅' if result['checkpoint_exists'] else '❌'}")
    else:
        print(f"    Checkpoint: ❌ WARNING: Not found")
    print(f"    F1 Macro: {result['test_f1_macro']:.4f}")
    print("")

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Map model names for better readability
df["model"] = df["model"].map({
    "imagenet_to_surgvu": "ImageNet",
    "moco_to_surgvu": "MoCo",
    "simclr_to_surgvu": "SimCLR"
})

# Map data subsets to percentages
df["data_subset"] = df["data_subset"].map({
    "12": "12.5%",
    "25": "25%",
    "100": "100%"
})

# Print detailed results
print("\nDetailed Results:")
print(df.to_string(index=False))

# Save the DataFrame to a CSV file
df.to_csv("train_val_results.csv", index=False)
print("\nResults saved to train_val_results.csv")