import os
import time
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from eval.metrics_utils import calculate_metrics, calculate_mse
from utils.gpu_utils import get_gpu_memory_used
from models.ml_models import train_rf, train_xgb, train_lgbm, train_linear

def train_ml_model(
    config: dict,
    Xh_train: np.ndarray,
    Xf_train: np.ndarray,
    y_train: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    y_test: np.ndarray,
    dates_test: list,
    scaler_target=None
):
    """
    Train a traditional ML model and evaluate on the test set.
    """

    def flatten(Xh, Xf):
        """
        简单的特征展平，保持DL和ML模型特征一致性 | Simple feature flattening, maintain feature consistency between DL and ML models
        """
        h = Xh.reshape(Xh.shape[0], -1)
        if Xf is not None:
            f = Xf.reshape(Xf.shape[0], -1)
            return np.concatenate([h, f], axis=1)
        return h

    X_train_flat = flatten(Xh_train, Xf_train)
    X_test_flat  = flatten(Xh_test, Xf_test)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat  = y_test.reshape(y_test.shape[0], -1)

    name = config['model']

    ml_param_keys = {
        'RF':     ['n_estimators', 'max_depth', 'random_state'],
        'XGB':    ['n_estimators', 'max_depth', 'learning_rate', 'verbosity'],
        'LGBM':   ['n_estimators', 'max_depth', 'learning_rate', 'random_state'],
        'Linear': []  # Linear Regression has no hyperparameters
    }
    # 根据模型复杂度选择参数 | Select parameters based on model complexity
    complexity = config.get('model_complexity', 'low')
    all_params = config.get('model_params', {})
    
    # 选择对应复杂度的参数 | Select parameters for corresponding complexity
    if complexity == 'high' and 'ml_high' in all_params:
        model_params = all_params['ml_high']
    elif complexity == 'low' and 'ml_low' in all_params:
        model_params = all_params['ml_low']
    else:
        # 回退到直接使用model_params中的参数 | Fall back to use parameters directly from model_params
        model_params = all_params
    
    allowed_keys = ml_param_keys.get(name, [])
    params = {k: model_params[k] for k in allowed_keys if k in model_params}

    if 'learning_rate' in params:
        params['learning_rate'] = float(params['learning_rate'])
    if 'n_estimators' in params:
        params['n_estimators'] = int(params['n_estimators'])
    if 'max_depth' in params and params['max_depth'] is not None:
        params['max_depth'] = int(params['max_depth'])
    if 'random_state' in params:
        params['random_state'] = int(params['random_state'])
    if 'verbosity' in params:
        params['verbosity'] = int(params['verbosity'])

    if name == 'RF':
        trainer = train_rf
    elif name == 'XGB':
        trainer = train_xgb
    elif name == 'LGBM':
        trainer = train_lgbm
    elif name == 'Linear':
        trainer = train_linear
    else:
        raise ValueError(f"Unsupported ML model: {name}")

    start_time = time.time()
    model = trainer(X_train_flat, y_train_flat, params)
    train_time = time.time() - start_time

    # Measure inference time
    inference_start = time.time()
    preds_flat = model.predict(X_test_flat)
    inference_time = time.time() - inference_start
    train_preds_flat = model.predict(X_train_flat)

    # 使用scaler_target进行逆变换 | Inverse transform using scaler_target
    fh = int(config['future_hours'])
    if scaler_target is not None:
        y_matrix = scaler_target.inverse_transform(y_test_flat).reshape(-1, fh)
        p_matrix = scaler_target.inverse_transform(preds_flat.reshape(-1, 1)).reshape(-1, fh)
    else:
        y_matrix = y_test_flat.reshape(-1, fh)
        p_matrix = preds_flat.reshape(-1, fh)
    
    # 裁剪预测值到合理范围[0, 100]（容量因子百分比） | Clip predictions to reasonable range [0, 100] (capacity factor percentage)
    p_matrix = np.clip(p_matrix, 0, 100)

    # === 计算所有评估指标 === | Calculate all evaluation metrics
    # 重要：指标已经基于逆变换后的真实值计算（容量因子百分比，0-100） | Important: metrics calculated based on inverse-transformed values (capacity factor percentage, 0-100)
    # 计算MSE | Calculate MSE
    mse = calculate_mse(y_matrix, p_matrix)
    
    # 计算所有指标 | Calculate all metrics
    all_metrics = calculate_metrics(y_matrix, p_matrix)
    
    # 提取基本指标 | Extract basic metrics
    rmse = all_metrics['rmse']
    mae = all_metrics['mae']
    
    train_mse = mean_squared_error(y_train_flat, train_preds_flat)

    # 获取GPU内存使用量 | Get GPU memory usage
    gpu_memory_used = get_gpu_memory_used()

    save_dir  = config['save_dir']
    
    # 根据配置决定是否保存模型 | Decide whether to save model based on config
    save_options = config.get('save_options', {})
    if save_options.get('save_model', False):
        model_dir = os.path.join(save_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))

    metrics = {
        'mse':            mse,
        'rmse':           rmse,
        'mae':            mae,
        'nrmse':          all_metrics['nrmse'],
        'r_square':       all_metrics['r_square'],
        'r2':             all_metrics['r2'],  # 添加r2别名 | Add r2 alias
        'smape':          all_metrics['smape'],
        'best_epoch':     np.nan,  # ML模型没有epoch概念 | ML models don't have epoch concept
        'final_lr':       np.nan,  # ML模型没有学习率概念 | ML models don't have learning rate concept
        'gpu_memory_used': gpu_memory_used,
        'train_time_sec': round(train_time, 2),
        'inference_time_sec': round(inference_time, 2),
        'param_count':    X_train_flat.shape[1],
        'samples_count':  len(y_matrix),
        'predictions':    p_matrix,
        'y_true':         y_matrix,
        'dates':          dates_test,
        'epoch_logs':     [{'epoch': 1, 'train_loss': train_mse, 'val_loss': mse}],
        'inverse_transformed': False  # Capacity Factor不需要逆标准化 | Capacity Factor doesn't need inverse normalization
    }

    return model, metrics
