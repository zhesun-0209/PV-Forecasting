"""
Test cuML GPU availability and diagnose issues
"""
import sys
import os

print("="*60)
print("Diagnosing cuML GPU availability")
print("="*60)

# Step 1: Check if cuML is installed
print("\n[1] Checking cuML installation...")
try:
    import cuml
    print(f"  ✓ cuML version: {cuml.__version__}")
except ImportError as e:
    print(f"  ✗ cuML not installed: {e}")
    print("\n  To install: pip install cuml-cu11")
    sys.exit(1)

# Step 2: Check cuDF
print("\n[2] Checking cuDF...")
try:
    import cudf
    print(f"  ✓ cuDF version: {cudf.__version__}")
except ImportError as e:
    print(f"  ✗ cuDF not installed: {e}")
    print("\n  cuDF is required for cuML GPU")
    sys.exit(1)

# Step 3: Check CuPy
print("\n[3] Checking CuPy...")
try:
    import cupy as cp
    print(f"  ✓ CuPy version: {cp.__version__}")
    test = cp.array([1, 2, 3])
    print(f"  ✓ CuPy GPU working: {test.device}")
except Exception as e:
    print(f"  ✗ CuPy error: {e}")
    sys.exit(1)

# Step 4: Check CUDA bindings
print("\n[4] Checking CUDA bindings...")
try:
    import cuda.bindings.cyruntime
    print("  ✓ cuda.bindings.cyruntime available")
except ImportError as e:
    print(f"  ✗ cuda.bindings not available: {e}")
    print("\n  To install: pip install cuda-bindings")

# Step 5: Test cuML Random Forest
print("\n[5] Testing cuML Random Forest...")
try:
    from cuml.ensemble import RandomForestRegressor
    import numpy as np
    
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    
    # Convert to cuDF (CRITICAL!)
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)
    
    rf = RandomForestRegressor(n_estimators=10, max_depth=5)
    rf.fit(X_cudf, y_cudf)
    pred = rf.predict(X_cudf[:10])
    
    print(f"  ✓ Random Forest GPU working (pred shape: {pred.shape})")
    
except Exception as e:
    print(f"  ✗ Random Forest failed: {e}")
    import traceback
    traceback.print_exc()

# Step 6: Test cuML Linear Regression
print("\n[6] Testing cuML Linear Regression...")
try:
    from cuml.linear_model import LinearRegression
    
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    
    # Convert to cuDF
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)
    
    lr = LinearRegression()
    lr.fit(X_cudf, y_cudf)
    pred = lr.predict(X_cudf[:10])
    
    print(f"  ✓ Linear Regression GPU working (pred shape: {pred.shape})")
    
except Exception as e:
    print(f"  ✗ Linear Regression failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Diagnosis complete!")
print("="*60)

