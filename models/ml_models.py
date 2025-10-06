"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
import numpy as np
import torch  # 添加torch导入用于GPU检测

try:
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.linear_model import LinearRegression as cuLinearRegression
    GPU_AVAILABLE = True
    print("✅ cuML RandomForestRegressor 和 LinearRegression 可用")  # cuML RandomForestRegressor and LinearRegression available
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from sklearn.linear_model import LinearRegression as cuLinearRegression
    GPU_AVAILABLE = False
    print("Warning: cuML not available, falling back to CPU versions")

# 检查XGBoost GPU支持 | Check XGBoost GPU support
XGB_GPU_AVAILABLE = False
try:
    import xgboost as xgb
    # 测试XGBoost GPU支持 | Test XGBoost GPU support
    try:
        test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        XGB_GPU_AVAILABLE = True
        print("XGBoost GPU 可用")  # XGBoost GPU available
    except:
        print("XGBoost GPU 不可用，使用CPU版本")  # XGBoost GPU unavailable, using CPU version
except ImportError:
    print("XGBoost 不可用")  # XGBoost unavailable

# 检查LightGBM GPU支持 | Check LightGBM GPU support
LGB_GPU_AVAILABLE = False
try:
    import lightgbm as lgb
    import numpy as np
    from sklearn.multioutput import MultiOutputRegressor
    # 测试LightGBM GPU支持 | Test LightGBM GPU support
    try:
        # 创建测试数据 | Create test data
        X_test = np.random.rand(10, 5)
        y_test = np.random.rand(10, 24)
        
        # 测试GPU训练 | Test GPU training
        base = lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=1, verbose=-1)
        model = MultiOutputRegressor(base)
        model.fit(X_test, y_test)
        LGB_GPU_AVAILABLE = True
        print("LightGBM GPU 可用")  # LightGBM GPU available
    except Exception as e:
        if "GPU Tree Learner was not enabled" in str(e):
            print("LightGBM GPU 不可用 - 需要重新编译支持GPU")  # LightGBM GPU unavailable - needs recompilation with GPU support
        else:
            print(f"LightGBM GPU 不可用: {e}")  # LightGBM GPU unavailable: {e}
except ImportError:
    print("LightGBM 不可用")  # LightGBM unavailable

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import threading

# GPU资源锁定，防止并行训练时的冲突 | GPU resource lock to prevent conflicts during parallel training
gpu_lock = threading.Lock()

def train_rf(X_train, y_train, params: dict):
    """Train Random Forest regressor with multi-output support - CPU fallback."""
    # 使用CPU版本的Random Forest，因为cuML在Windows上安装困难 | Use CPU version of Random Forest as cuML is difficult to install on Windows
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    try:
        # 检查数据有效性 | Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用sklearn Random Forest CPU版本 | Use sklearn Random Forest CPU version
        rf_params = {
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 10),
            'random_state': 42
        }
        base = RandomForestRegressor(**rf_params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Random Forest训练失败: {e}")  # Random Forest training failed: {e}
        raise RuntimeError(f"Random Forest训练失败: {e}")

# GBR已移除，使用XGBoost和LightGBM替代 | GBR removed, use XGBoost and LightGBM instead

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support - GPU preferred."""
    try:
        # 检查数据有效性 | Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 尝试使用GPU，如果不可用则使用CPU | Try to use GPU, if unavailable use CPU
        if torch.cuda.is_available():
            print("使用XGBoost GPU版本")  # Using XGBoost GPU version
            gpu_params = params.copy()
            gpu_params.update({
                'tree_method': 'hist',
                'device': 'cuda',
                'verbosity': 0
            })
            base = XGBRegressor(**gpu_params)
        else:
            print("GPU不可用，使用XGBoost CPU版本")  # GPU unavailable, using XGBoost CPU version
            base = XGBRegressor(**params)
        
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"XGBoost训练失败: {e}")  # XGBoost training failed: {e}
        raise RuntimeError(f"XGBoost训练失败: {e}")

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support - GPU preferred."""
    try:
        # 检查数据有效性 | Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 尝试使用GPU，如果不可用则使用CPU | Try to use GPU, if unavailable use CPU
        if torch.cuda.is_available():
            print("使用LightGBM GPU版本")  # Using LightGBM GPU version
            gpu_params = params.copy()
            gpu_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1
            })
            base = LGBMRegressor(**gpu_params)
        else:
            print("GPU不可用，使用LightGBM CPU版本")  # GPU unavailable, using LightGBM CPU version
            base = LGBMRegressor(**params)
        
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"LightGBM训练失败: {e}")  # LightGBM training failed: {e}
        raise RuntimeError(f"LightGBM训练失败: {e}")

def train_linear(X_train, y_train, params: dict):
    """Train Linear Regression with GPU support if available."""
    with gpu_lock:  # 确保GPU资源独占 | Ensure exclusive GPU resource access
        try:
            # 检查数据有效性 | Check data validity
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                print("检测到NaN或Inf值，进行清理")  # Detected NaN or Inf values, cleaning
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if GPU_AVAILABLE:
                # 使用cuML GPU版本 | Use cuML GPU version
                model = cuLinearRegression()
                model.fit(X_train, y_train)
                return model
            else:
                # 使用sklearn CPU版本 | Use sklearn CPU version
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                return model
        except Exception as e:
            print(f"Linear Regression训练失败: {e}")  # Linear Regression training failed: {e}
            raise RuntimeError(f"Linear Regression训练失败: {e}")
