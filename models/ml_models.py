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
    # Test XGBoost GPU support
    try:
        test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        XGB_GPU_AVAILABLE = True
        print("XGBoost GPU available")  # XGBoost GPU available
    except:
        print("XGBoost GPU unavailable, using CPU version")  # XGBoost GPU unavailable, using CPU version
except ImportError:
    print("XGBoost unavailable")  # XGBoost unavailable

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
        
        rf_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 10),
            'random_state': 42
        }
        
        # Try GPU version with cuDF conversion
        if GPU_RF_AVAILABLE:
            try:
                import cudf
                import cupy as cp
                from sklearn.multioutput import MultiOutputRegressor
                
                print("Using Random Forest GPU (cuML)")
                
                # Convert to cuDF for cuML (key step!)
                X_cudf = cudf.DataFrame(X_train)
                y_cudf = cudf.DataFrame(y_train)
                
                base = cuRandomForestRegressor(**rf_params)
                model = MultiOutputRegressor(base)
                model.fit(X_cudf, y_cudf)
                return model
            except Exception as e:
                print(f"cuML GPU failed ({e}), falling back to sklearn CPU")
                # Fall through to CPU version
        
        # CPU version (sklearn)
        print("Using Random Forest CPU (sklearn)")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        base = RandomForestRegressor(**rf_params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
        
    except Exception as e:
        print(f"Random Forest training failed: {e}")
        raise RuntimeError(f"Random Forest training failed: {e}")

# GBR removed, use XGBoost and LightGBM instead

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support - GPU preferred."""
    try:
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")  # Detected NaN or Inf values, cleaning
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")  # Detected NaN or Inf values, cleaning
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Try to use GPU, if unavailable use CPU
        if torch.cuda.is_available():
            print("Using XGBoost GPU version")  # Using XGBoost GPU version
            gpu_params = params.copy()
            gpu_params.update({
                'tree_method': 'hist',
                'device': 'cuda',
                'verbosity': 0
            })
            base = XGBRegressor(**gpu_params)
        else:
            print("GPU unavailable, using XGBoost CPU version")  # GPU unavailable, using XGBoost CPU version
            base = XGBRegressor(**params)
        
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"XGBoost training failed: {e}")  # XGBoost training failed: {e}
        raise RuntimeError(f"XGBoost training failed: {e}")

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support - GPU preferred."""
    try:
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")  # Detected NaN or Inf values, cleaning
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")  # Detected NaN or Inf values, cleaning
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Try to use GPU, if unavailable use CPU
        if torch.cuda.is_available():
            print("Using LightGBM GPU version")  # Using LightGBM GPU version
            gpu_params = params.copy()
            gpu_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1
            })
            base = LGBMRegressor(**gpu_params)
        else:
            print("GPU unavailable, using LightGBM CPU version")  # GPU unavailable, using LightGBM CPU version
            base = LGBMRegressor(**params)
        
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"LightGBM training failed: {e}")  # LightGBM training failed: {e}
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
            
            # Try GPU version with cuDF conversion
            if GPU_LINEAR_AVAILABLE:
                try:
                    import cudf
                    
                    print("Using Linear Regression GPU (cuML)")
                    
                    # Convert to cuDF for cuML (key step!)
                    X_cudf = cudf.DataFrame(X_train)
                    y_cudf = cudf.DataFrame(y_train)
                    
                    model = cuLinearRegression()
                    model.fit(X_cudf, y_cudf)
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
