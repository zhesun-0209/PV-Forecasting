#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save experiment results to Google Drive
Automatically copy all result files to specified Drive location
"""

import os
import shutil
import glob
from datetime import datetime
from pathlib import Path


def save_to_drive(drive_path='/content/drive/MyDrive/Solar PV electricity/results',
                  local_results_dir='.',
                  include_detailed_results=False):
    """
    Save all experiment results to Google Drive
    
    Args:
        drive_path: Target path in Google Drive
        local_results_dir: Local directory containing results
        include_detailed_results: Whether to include results/ directory (can be large)
    """
    print("=" * 80)
    print("Saving Results to Google Drive")
    print("=" * 80)
    
    # Check if running in Colab
    try:
        from google.colab import drive
        is_colab = True
    except ImportError:
        is_colab = False
        print("[WARNING] Not running in Colab, Google Drive functions unavailable")
        print("Saving to local directory instead...")
        drive_path = './saved_results'
    
    # Mount Drive if in Colab and not already mounted
    if is_colab:
        if not os.path.exists('/content/drive'):
            print("\nMounting Google Drive...")
            drive.mount('/content/drive')
            print("[OK] Google Drive mounted")
        else:
            print("[OK] Google Drive already mounted")
    
    # Create target directory
    os.makedirs(drive_path, exist_ok=True)
    print(f"\nTarget directory: {drive_path}")
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(drive_path, f'session_{timestamp}')
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session directory: {session_dir}")
    
    # Find all result CSV files
    result_patterns = [
        'results_*.csv',
        'all_experiments_results_*.csv',
        'all_plants_status_report.csv'
    ]
    
    copied_files = []
    
    print("\n" + "-" * 80)
    print("Copying result files...")
    print("-" * 80)
    
    for pattern in result_patterns:
        files = glob.glob(os.path.join(local_results_dir, pattern))
        for file in files:
            filename = os.path.basename(file)
            dest = os.path.join(session_dir, filename)
            
            try:
                shutil.copy2(file, dest)
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"  [OK] Copied: {filename} ({file_size:.1f} KB)")
                copied_files.append(filename)
            except Exception as e:
                print(f"  [ERROR] Failed to copy {filename}: {str(e)}")
    
    # Copy detailed results directory if requested
    if include_detailed_results:
        results_dir = os.path.join(local_results_dir, 'results')
        if os.path.exists(results_dir):
            print("\n" + "-" * 80)
            print("Copying detailed results directory...")
            print("-" * 80)
            
            dest_results = os.path.join(session_dir, 'results')
            try:
                shutil.copytree(results_dir, dest_results, dirs_exist_ok=True)
                print(f"  [OK] Copied results/ directory")
                
                # Calculate total size
                total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, _, filenames in os.walk(dest_results)
                               for filename in filenames)
                print(f"  Size: {total_size / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"  [ERROR] Failed to copy results/ directory: {str(e)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Save Summary")
    print("=" * 80)
    print(f"Files copied: {len(copied_files)}")
    print(f"Destination: {session_dir}")
    
    if copied_files:
        print(f"\nCopied files:")
        for f in copied_files:
            print(f"  - {f}")
    
    print("\n[OK] Results saved to Google Drive successfully!")
    print("=" * 80)
    
    return session_dir


def save_specific_plant(plant_id, drive_path='/content/drive/MyDrive/Solar PV electricity/results'):
    """
    Save results for a specific plant
    
    Args:
        plant_id: Plant ID (e.g., '1140')
        drive_path: Target path in Google Drive
    """
    print(f"Saving results for Plant {plant_id}...")
    
    result_files = glob.glob(f'results_{plant_id}_*.csv')
    
    if not result_files:
        print(f"[WARNING] No result files found for Plant {plant_id}")
        return
    
    # Mount Drive if in Colab
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
    except:
        pass
    
    # Create directory
    os.makedirs(drive_path, exist_ok=True)
    
    # Copy files
    for file in result_files:
        dest = os.path.join(drive_path, os.path.basename(file))
        shutil.copy2(file, dest)
        print(f"  [OK] Copied: {os.path.basename(file)}")
    
    print(f"[OK] Plant {plant_id} results saved to {drive_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Save experiment results to Google Drive')
    parser.add_argument('--drive-path', 
                       default='/content/drive/MyDrive/Solar PV electricity/results',
                       help='Google Drive target path')
    parser.add_argument('--include-detailed', action='store_true',
                       help='Include detailed results/ directory (can be large)')
    parser.add_argument('--plant', type=str,
                       help='Save results for specific plant only')
    
    args = parser.parse_args()
    
    if args.plant:
        save_specific_plant(args.plant, args.drive_path)
    else:
        save_to_drive(args.drive_path, 
                     include_detailed_results=args.include_detailed)

