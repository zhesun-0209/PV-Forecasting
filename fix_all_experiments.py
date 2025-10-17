#!/usr/bin/env python3
"""
Quick fix script to update all sensitivity analysis experiments
to use the corrected run_single_experiment() function
"""

import os
import re

# All experiment files to fix
experiment_files = [
    'sensitivity_analysis/hourly_effect.py',
    'sensitivity_analysis/weather_feature_adoption.py',
    'sensitivity_analysis/lookback_window.py',
    'sensitivity_analysis/model_complexity.py',
    'sensitivity_analysis/training_scale.py',
    'sensitivity_analysis/no_shuffle.py',
    'sensitivity_analysis/dataset_extension.py'
]

# The old pattern to find (simplified version)
old_pattern = r'try:\s+#\s*Preprocess data.*?result\.get\(\'train_time\',\s*0\)'

# New code block (will be customized for each file)
new_code_template = """try:
                # Run experiment using the corrected function
                result = run_single_experiment(config, df.copy(), use_sliding_windows={use_sliding})
                
                # Check if experiment succeeded
                if result['status'] != 'SUCCESS':
                    print(f"  Error running {{model}}: {{result.get('error', 'Unknown error')}}")
                    continue
                
                # Get metrics from result
                mae = result.get('mae', np.nan)
                rmse = result.get('rmse', np.nan)
                r2 = result.get('r2', np.nan)
                nrmse = result.get('nrmse', np.nan) or compute_nrmse(result.get('y_test').flatten(), result.get('y_test_pred').flatten())
                train_time = result.get('train_time', 0)"""

print("Fixing sensitivity analysis experiments...")
print("=" * 80)

for filepath in experiment_files:
    print(f"\nProcessing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  [SKIP] File not found")
        continue
    
    # Determine if this file uses sliding windows
    use_sliding = 'False'
    if 'dataset_extension' in filepath:
        use_sliding = 'True'
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'run_single_experiment(config, df.copy()' in content:
        print(f"  [SKIP] Already fixed")
        continue
    
    # Manual fix for each file type
    # This is a simplified approach - manually identify the section to replace
    
    # For most experiments, find the training section
    lines = content.split('\n')
    new_lines = []
    in_training_block = False
    skip_until_except = False
    
    for i, line in enumerate(lines):
        if 'try:' in line and ('Preprocess data' in lines[i+1] if i+1 < len(lines) else False):
            # Start of training block
            in_training_block = True
            skip_until_except = True
            # Add the new code
            indent = len(line) - len(line.lstrip())
            new_code = new_code_template.format(use_sliding=use_sliding)
            # Adjust indentation
            new_code_lines = new_code.split('\n')
            for nc_line in new_code_lines:
                if nc_line.strip():
                    new_lines.append(' ' * indent + nc_line.lstrip())
                else:
                    new_lines.append(nc_line)
            continue
        
        if skip_until_except:
            if 'except Exception as e:' in line:
                skip_until_except = False
                in_training_block = False
            else:
                continue  # Skip old code
        
        new_lines.append(line)
    
    # Write back
    new_content = '\n'.join(new_lines)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  [OK] Fixed")

print("\n" + "=" * 80)
print("All experiments fixed!")
print("\nPlease review the changes and test the experiments.")

