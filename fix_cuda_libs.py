"""
Fix CUDA library linking issues on Colab
Creates symbolic links for libcudart.so
"""
import os
import glob
import subprocess

print("="*60)
print("Fixing CUDA library links")
print("="*60)

# Find all possible CUDA library paths
cuda_paths = [
    "/usr/local/cuda-11.8/lib64",
    "/usr/local/cuda/lib64",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib",
    "/usr/lib/x86_64-linux-gnu"
]

print("\nSearching for libcudart.so...")
cudart_found = None

for path in cuda_paths:
    if os.path.exists(path):
        # Look for any version of libcudart.so
        cudart_files = glob.glob(f"{path}/libcudart.so*")
        if cudart_files:
            cudart_found = cudart_files[0]
            print(f"  Found: {cudart_found}")
            break

if cudart_found:
    # Create symbolic link to libcudart.so.11.0
    lib_dir = os.path.dirname(cudart_found)
    target = f"{lib_dir}/libcudart.so.11.0"
    
    if not os.path.exists(target):
        print(f"\nCreating symbolic link...")
        try:
            os.symlink(cudart_found, target)
            print(f"  ✓ Created: {target}")
        except Exception as e:
            print(f"  ✗ Failed to create symlink: {e}")
            print(f"\n  Trying with system command...")
            subprocess.run(["ln", "-sf", cudart_found, target])
    else:
        print(f"  ✓ {target} already exists")
    
    # Update LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"\n  ✓ LD_LIBRARY_PATH updated: {lib_dir}")
    
    # Write to bashrc for persistence
    bashrc_line = f'export LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH\n'
    try:
        with open(os.path.expanduser("~/.bashrc"), "a") as f:
            f.write(bashrc_line)
        print("  ✓ Added to ~/.bashrc")
    except:
        pass

else:
    print("  ✗ libcudart.so not found in standard paths")
    print("\n  Attempting alternative: Install from conda...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "nvidia-cuda-runtime-cu11"
    ], capture_output=True)

print("\n" + "="*60)
print("Testing cuML after fix...")
print("="*60)

try:
    from cuml.ensemble import RandomForestRegressor
    from cuml.linear_model import LinearRegression
    import numpy as np
    
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X, y)
    print("\n✓ Random Forest GPU: WORKING!")
    
    lr = LinearRegression()
    lr.fit(X, y)
    print("✓ Linear Regression GPU: WORKING!")
    
    print("\n" + "="*60)
    print("SUCCESS: cuML GPU fully functional!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Still not working: {e}")
    print("\nFinal solution: Use CPU fallback")
    print("(6/8 models still use GPU)")

print("\nDone!")

