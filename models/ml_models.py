"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
import numpy as np
import torch

# Try to import cuML for GPU-accelerated Random Forest and Linear Regression
GPU_RF_AVAILABLE = False
GPU_LINEAR_AVAILABLE = False

try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.linear_model import LinearRegression as cuLinearRegression
    GPU_RF_AVAILABLE = True
    GPU_LINEAR_AVAILABLE = True
    print("cuML available (GPU-accelerated Random Forest and Linear Regression)")
except Exception:
    # Silently fallback to sklearn
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from sklearn.linear_model import LinearRegression as cuLinearRegression
    GPU_RF_AVAILABLE = False
    GPU_LINEAR_AVAILABLE = False
    print("Using sklearn (CPU) for Random Forest and Linear Regression")

# Check XGBoost GPU support
XGB_GPU_AVAILABLE = False
try:
    import xgboost as xgb
    # Test XGBoost GPU support with proper GPU method
    try:
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=1)
        XGB_GPU_AVAILABLE = True
        print("XGBoost GPU available")
    except:
        print("XGBoost GPU unavailable, using CPU version")
except ImportError:
    print("XGBoost unavailable")

# Check LightGBM GPU support
LGB_GPU_AVAILABLE = False
try:
    import lightgbm as lgb
    import numpy as np
    from sklearn.multioutput import MultiOutputRegressor
    # Test LightGBM GPU support
    try:
        # Create test data
        X_test = np.random.rand(10, 5)
        y_test = np.random.rand(10, 24)
        
        # Test GPU training
        base = lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=1, verbose=-1)
        model = MultiOutputRegressor(base)
        model.fit(X_test, y_test)
        LGB_GPU_AVAILABLE = True
        print("LightGBM GPU available")  # LightGBM GPU available
    except Exception as e:
        if "GPU Tree Learner was not enabled" in str(e):
            print("LightGBM GPU unavailable - needs recompilation with GPU support")  # LightGBM GPU unavailable - needs recompilation with GPU support
        else:
            print(f"LightGBM GPU unavailable: {e}")  # LightGBM GPU unavailable: {e}
except ImportError:
    print("LightGBM unavailable")  # LightGBM unavailable

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import threading

# GPU resource lock to prevent conflicts during parallel training
gpu_lock = threading.Lock()

def train_rf(X_train, y_train, params: dict):
    """Train Random Forest regressor with GPU support via cuML."""
    try:
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        n_est = params.get('n_estimators', 100)
        max_d = params.get('max_depth', 10)
        
        rf_params = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'random_state': 42,
            'n_jobs': -1  # Use all CPU cores for sklearn
        }
        
        print(f"Training RF: n_estimators={n_est}, max_depth={max_d}, samples={len(X_train)}, features={X_train.shape[1]}")
        
        # Try GPU version with direct NumPy arrays (cuML compatibility)
        if GPU_RF_AVAILABLE:
            try:
                from sklearn.multioutput import MultiOutputRegressor
                import signal
                
                print("Attempting Random Forest GPU (cuML)...")
                
                # Set alarm for timeout (not available on Windows, skip on Windows)
                try:
                    # This will only work on Unix/Linux (Colab)
                    def timeout_handler(signum, frame):
                        raise TimeoutError("RF GPU training timeout")
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(600)  # 10 minutes timeout
                except (AttributeError, ValueError):
                    # Windows doesn't support SIGALRM, skip timeout
                    pass
                
                # cuML works directly with NumPy arrays (no cuDF needed!)
                base = cuRandomForestRegressor(**rf_params)
                model = MultiOutputRegressor(base, n_jobs=1)
                model.fit(X_train, y_train)
                
                try:
                    signal.alarm(0)  # Cancel alarm
                except (AttributeError, ValueError):
                    pass
                
                print("✓ Random Forest GPU (cuML) training successful")
                return model
                
            except (Exception, TimeoutError) as e:
                print(f"✗ cuML GPU failed or timeout ({type(e).__name__}: {e})")
                print("  Falling back to sklearn CPU (this is safe and expected)")
                try:
                    signal.alarm(0)  # Cancel alarm if set
                except (AttributeError, ValueError):
                    pass
                # Fall through to CPU version
        
        # CPU version (sklearn) - Always works
        print("Using Random Forest CPU (sklearn) - stable fallback")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        base = RandomForestRegressor(**rf_params)
        model = MultiOutputRegressor(base, n_jobs=1)  # Don't parallelize wrapper
        model.fit(X_train, y_train)
        print("✓ Random Forest CPU (sklearn) training successful")
        return model
        
    except Exception as e:
        print(f"✗ Random Forest training completely failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Random Forest training failed: {e}")

# GBR removed, use XGBoost and LightGBM instead

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support - GPU optimized."""
    try:
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Try to use GPU, if unavailable use CPU
        if torch.cuda.is_available() and XGB_GPU_AVAILABLE:
            print("Using XGBoost GPU version (gpu_hist)")
            gpu_params = params.copy()
            gpu_params.update({
                'tree_method': 'gpu_hist',  # GPU-optimized histogram algorithm
                'device': 'cuda',
                'verbosity': 0,
                'n_jobs': -1,  # Use all available CPU threads for data loading
                'predictor': 'gpu_predictor'  # GPU predictor for inference
            })
            base = XGBRegressor(**gpu_params)
        else:
            print("GPU unavailable, using XGBoost CPU version")
            cpu_params = params.copy()
            cpu_params.update({
                'tree_method': 'hist',  # CPU histogram method
                'n_jobs': -1,  # Use all CPU cores
                'verbosity': 0
            })
            base = XGBRegressor(**cpu_params)
        
        model = MultiOutputRegressor(base, n_jobs=1)  # Don't parallelize MultiOutputRegressor to avoid conflicts
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        raise RuntimeError(f"XGBoost training failed: {e}")

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support - GPU optimized."""
    try:
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Try to use GPU, if unavailable use CPU
        if torch.cuda.is_available() and LGB_GPU_AVAILABLE:
            print("Using LightGBM GPU version")
            gpu_params = params.copy()
            gpu_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1,
                'n_jobs': -1  # Use all CPU threads for data loading
            })
            base = LGBMRegressor(**gpu_params)
        else:
            print("GPU unavailable, using LightGBM CPU version")
            cpu_params = params.copy()
            cpu_params.update({
                'device': 'cpu',
                'n_jobs': -1,  # Use all CPU cores
                'verbose': -1
            })
            base = LGBMRegressor(**cpu_params)
        
        model = MultiOutputRegressor(base, n_jobs=1)  # Don't parallelize MultiOutputRegressor to avoid conflicts
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"LightGBM training failed: {e}")
        raise RuntimeError(f"LightGBM training failed: {e}")

def train_linear(X_train, y_train, params: dict):
    """Train Linear Regression with GPU support if available."""
    with gpu_lock:
        try:
            # Check data validity
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                print("Detected NaN or Inf values, cleaning")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                print("Detected NaN or Inf values, cleaning")
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Try GPU version with direct NumPy arrays (cuML compatibility)
            if GPU_LINEAR_AVAILABLE:
                try:
                    print("Using Linear Regression GPU (cuML)")
                    
                    # cuML works directly with NumPy arrays (no cuDF needed!)
                    model = cuLinearRegression()
                    model.fit(X_train, y_train)  # 直接使用NumPy数组
                    return model
                except Exception as e:
                    print(f"cuML GPU failed ({e}), falling back to sklearn CPU")
                    # Fall through to CPU version
            
            # CPU version (sklearn)
            print("Using Linear Regression CPU (sklearn)")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            print(f"Linear Regression training failed: {e}")
            raise RuntimeError(f"Linear Regression training failed: {e}")
