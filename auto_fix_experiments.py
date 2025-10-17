#!/usr/bin/env python3
"""
Automatically fix all sensitivity analysis experiment scripts
to use the corrected run_single_experiment() function
"""

import re

# Template for the fixed experiment running code
FIXED_CODE_TEMPLATE = """            try:
                # Run experiment using the corrected function
                result = run_single_experiment(config, df.copy(), use_sliding_windows={sliding})
                
                # Check if experiment succeeded
                if result['status'] != 'SUCCESS':
                    print(f"  Error running {{model}}: {{result.get('error', 'Unknown error')}}")
                    continue
                
                # Extract metrics
                mae = result['mae']
                rmse = result['rmse']
                r2 = result['r2']
                nrmse = result.get('nrmse', compute_nrmse(result['y_test'].flatten(), result['y_test_pred'].flatten()))
                train_time = result['train_time']
                test_samples = result['test_samples']
"""

def fix_file(filepath, use_sliding_windows=False):
    print(f"Fixing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'run_single_experiment(config, df.copy()' in content:
        print(f"  Already fixed, skipping")
        return False
    
    # Pattern to find the training block
    # Find from "try:" to "except Exception as e:"
    pattern = r'(            try:\s+#[^\n]+\n)(.*?)(            except Exception as e:)'
    
    def replacement(match):
        indent = match.group(1)
        old_code = match.group(2)
        except_line = match.group(3)
        
        # Generate new code
        new_code = FIXED_CODE_TEMPLATE.format(sliding='True' if use_sliding_windows else 'False')
        
        return indent.replace('# Preprocess data', '# Run experiment using corrected function') + new_code + '\n' + except_line
    
    # Apply replacement
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        print(f"  Warning: No changes made")
        return False
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  Fixed successfully")
    return True

# Fix all files
files_to_fix = [
    ('sensitivity_analysis/hourly_effect.py', False),
    ('sensitivity_analysis/weather_feature_adoption.py', False),
    ('sensitivity_analysis/lookback_window.py', False),
    ('sensitivity_analysis/model_complexity.py', False),
    ('sensitivity_analysis/training_scale.py', False),
    ('sensitivity_analysis/no_shuffle.py', False),
    ('sensitivity_analysis/dataset_extension.py', True),  # This one uses sliding windows
]

print("=" * 80)
print("Fixing sensitivity analysis experiment scripts")
print("=" * 80)

fixed_count = 0
for filepath, use_sliding in files_to_fix:
    if fix_file(filepath, use_sliding):
        fixed_count += 1

print("=" * 80)
print(f"Fixed {fixed_count} files")
print("=" * 80)

