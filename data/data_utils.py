# ======== data/data_utils.py ========

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 基于实际数据中的天气特征，简化为两种类别 | Based on actual weather features in data, simplified into two categories
# 太阳辐射特征 - 最重要的特征 | Solar irradiance features - most important features
IRRADIANCE_FEATURES = [
    'global_tilted_irradiance',    # 全球倾斜辐射 (最重要的辐射特征) | Global tilted irradiance (most important irradiance feature)
]

# 全部天气特征 - 去除低相关性特征，只保留高相关性特征 | All weather features - removed low correlation features, only keep high correlation features
ALL_WEATHER_FEATURES = [
    'global_tilted_irradiance',    # 全球倾斜辐射 (最重要) | Global tilted irradiance (most important)
    'vapour_pressure_deficit',     # 水汽压差 | Vapour pressure deficit
    'relative_humidity_2m',        # 相对湿度 | Relative humidity at 2m
    'temperature_2m',              # 温度 | Temperature at 2m
    'wind_gusts_10m',             # 10米阵风 | Wind gusts at 10m
    'cloud_cover_low',            # 低云覆盖 | Low cloud cover
    'wind_speed_100m',            # 100米风速 | Wind speed at 100m
    # 已移除低相关性特征: | Removed low correlation features:
    # 'snow_depth',               # 雪深 - 低相关性 | Snow depth - low correlation
    # 'dew_point_2m',             # 露点温度 - 低相关性 | Dew point at 2m - low correlation
    # 'surface_pressure',         # 表面气压 - 低相关性 | Surface pressure - low correlation
    # 'precipitation',            # 降水 - 低相关性 | Precipitation - low correlation
]

# 敏感性分析天气特征定义 | Weather feature definitions for sensitivity analysis
SOLAR_IRRADIANCE_FEATURES = ['global_tilted_irradiance']
HIGH_WEATHER_FEATURES = ['global_tilted_irradiance', 'vapour_pressure_deficit', 'relative_humidity_2m']
MEDIUM_WEATHER_FEATURES = HIGH_WEATHER_FEATURES + ['temperature_2m', 'wind_gusts_10m', 'cloud_cover_low', 'wind_speed_100m']
LOW_WEATHER_FEATURES = ALL_WEATHER_FEATURES  # 现在与ALL_WEATHER_FEATURES相同，因为已去除低相关性特征

# 根据天气特征类别选择特征 | Select features based on weather feature category
def get_weather_features_by_category(weather_category):
    """
    根据天气特征类别返回天气特征 | Return weather features based on category
    
    Args:
        weather_category: 'none', 'all_weather', 'solar_irradiance_only', 'high_weather', 'medium_weather', 'low_weather'
    
    Returns:
        list: 选中的天气特征列表 | List of selected weather features
    """
    if weather_category == 'none':
        return []  # 不返回任何天气特征 | Return no weather features
    elif weather_category == 'all_weather':
        return ALL_WEATHER_FEATURES
    elif weather_category == 'solar_irradiance_only':
        return SOLAR_IRRADIANCE_FEATURES
    elif weather_category == 'high_weather':
        return HIGH_WEATHER_FEATURES
    elif weather_category == 'medium_weather':
        return MEDIUM_WEATHER_FEATURES
    elif weather_category == 'low_weather':
        return LOW_WEATHER_FEATURES
    else:
        raise ValueError(f"Invalid weather_category: {weather_category}")

# 保持向后兼容性 | Maintain backward compatibility
BASE_HIST_FEATURES = IRRADIANCE_FEATURES
BASE_FCST_FEATURES = IRRADIANCE_FEATURES

# 时间编码特征 | Time encoding features
TIME_FEATURES = ['month_cos', 'month_sin', 'hour_cos', 'hour_sin']

TARGET_COL = 'Capacity Factor'

# 统计特征函数已移除 | Statistical feature functions removed

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    return df

def preprocess_features(df: pd.DataFrame, config: dict):
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # 日期过滤：只使用2022-01-01之后的数据 | Date filtering: only use data after 2022-01-01
    start_date = config.get('start_date', '2022-01-01')
    end_date = config.get('end_date', '2024-09-28')
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df_clean = df_clean[df_clean['Datetime'] >= start_dt].copy()
        print(f"过滤后数据（从{start_date}开始）: {len(df_clean)}行")  # Filtered data (starting from {start_date}): {len(df_clean)} rows
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df_clean = df_clean[df_clean['Datetime'] <= end_dt].copy()
        print(f"过滤后数据（到{end_date}结束）: {len(df_clean)}行")  # Filtered data (ending at {end_date}): {len(df_clean)} rows

    # 添加时间编码特征（根据开关决定） | Add time encoding features (based on switch)
    use_time_encoding = config.get('use_time_encoding', True)
    if use_time_encoding:
        df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['Month'] / 12)
        df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['Month'] / 12)
        df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['Hour'] / 24)
        df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['Hour'] / 24)

    # 构建特征列表 | Build feature lists
    hist_feats = []
    fcst_feats = []

    # 获取天气特征类别 | Get weather feature category
    weather_category = config.get('weather_category', 'none')

    # PV特征（历史发电量） | PV features (historical power generation)
    if config.get('use_pv', False):
        # 创建历史发电量特征：使用shift操作获取过去时间点的发电量 | Create historical power features: use shift operation to get past power generation
        # 这里不直接复制目标值，而是使用shift(1)获取前一个时间点的值 | Don't directly copy target values, use shift(1) to get previous time point values
        # 在滑动窗口中，这将提供真正的历史发电量数据 | In sliding windows, this will provide true historical power generation data
        df_clean['Capacity_Factor_hist'] = df_clean[TARGET_COL].shift(1)
        hist_feats.append('Capacity_Factor_hist')

    # 历史天气特征（HW）- 不带_pred后缀 | Historical weather features (HW) - without _pred suffix
    if config.get('use_hist_weather', False):
        hist_feats += get_weather_features_by_category(weather_category)

    # 时间编码特征（根据开关决定） | Time encoding features (based on switch)
    if use_time_encoding:
        hist_feats += TIME_FEATURES

    # 预测特征（NWP） | Forecast features (NWP)
    if config.get('use_forecast', False):
        if config.get('use_ideal_nwp', False):
            # 理想NWP+：使用目标日的HW特征（不带_pred后缀） | Ideal NWP+: use target day HW features (without _pred suffix)
            base_weather_features = get_weather_features_by_category(weather_category)
            fcst_feats += base_weather_features
            # 添加时间编码特征到预测特征 | Add time encoding features to forecast features
            if use_time_encoding:
                fcst_feats += TIME_FEATURES
        else:
            # 普通NWP：使用带_pred后缀的预测特征 | Normal NWP: use forecast features with _pred suffix
            base_weather_features = get_weather_features_by_category(weather_category)
            forecast_features = [f + '_pred' for f in base_weather_features]
            fcst_feats += forecast_features
            # 添加时间编码特征到预测特征 | Add time encoding features to forecast features
            if use_time_encoding:
                fcst_feats += TIME_FEATURES

    # 检查是否为纯NWP模式（无历史数据） | Check if pure NWP mode (no historical data)
    no_hist_power = config.get('no_hist_power', False)
    # 只有在用户没有明确设置no_hist_power时才自动检测 | Auto-detect only when user hasn't explicitly set no_hist_power
    if 'no_hist_power' not in config:
        # 自动检测：如果既没有历史发电也没有历史天气，则为纯NWP模式 | Auto-detect: if neither historical power nor historical weather, then pure NWP mode
        no_hist_power = not config.get('use_pv', False) and not config.get('use_hist_weather', False)

    # 确保所有特征都存在
    available_hist_feats = [f for f in hist_feats if f in df_clean.columns]
    available_fcst_feats = [f for f in fcst_feats if f in df_clean.columns]

    # 删除缺失值（但保留历史特征中的NaN，因为shift操作会产生NaN）
    na_check_feats = available_fcst_feats + [TARGET_COL]  # 不检查历史特征
    df_clean = df_clean.dropna(subset=na_check_feats).reset_index(drop=True)
    
    # 对于历史特征，将NaN替换为0（表示没有历史数据）
    for feat in available_hist_feats:
        df_clean[feat] = df_clean[feat].fillna(0.0)

    # 标准化特征
    scaler_hist = MinMaxScaler()
    if available_hist_feats:
        # 检查特征是否有足够的变异性
        for feat in available_hist_feats:
            if df_clean[feat].std() == 0:
                print(f"特征 {feat} 标准差为0，添加微小噪声避免除零错误")
                df_clean[feat] += np.random.normal(0, 1e-8, len(df_clean))
        df_clean[available_hist_feats] = scaler_hist.fit_transform(df_clean[available_hist_feats])

    scaler_fcst = MinMaxScaler()
    if available_fcst_feats:
        # 检查特征是否有足够的变异性
        for feat in available_fcst_feats:
            if df_clean[feat].std() == 0:
                print(f"特征 {feat} 标准差为0，添加微小噪声避免除零错误")
                df_clean[feat] += np.random.normal(0, 1e-8, len(df_clean))
        df_clean[available_fcst_feats] = scaler_fcst.fit_transform(df_clean[available_fcst_feats])

    # 标准化目标变量（Capacity Factor）
    # 虽然范围是0-67.76，但标准化有助于训练稳定性
    scaler_target = MinMaxScaler()
    df_clean[TARGET_COL] = scaler_target.fit_transform(df_clean[[TARGET_COL]]).flatten()

    df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)

    # 将no_hist_power参数添加到返回的元组中，供调用者使用
    return df_clean, available_hist_feats, available_fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power

def create_sliding_windows(df, past_hours, future_hours, hist_feats, fcst_feats, no_hist_power=False):
    """
    创建连续时间序列滑动窗口数据
    
    Args:
        df: 预处理后的数据框
        past_hours: 历史时间窗口（小时）
        future_hours: 预测时间窗口（小时）
        hist_feats: 历史特征列表
        fcst_feats: 预测特征列表
        no_hist_power: 如果为True，不使用历史发电量数据，只使用预测天气
    """
    X_hist, y, hours, dates = [], [], [], []
    X_fcst = [] if fcst_feats else None  # 只有在需要预测特征时才初始化
    n = len(df)
    
    # 连续时间序列方法：直接使用时间序列数据，不按天分组
    # 为每个时间点创建样本
    for i in range(past_hours, n - future_hours + 1):
        # 历史数据：前past_hours小时
        hist_start = i - past_hours
        hist_end = i
        hist_data = df.iloc[hist_start:hist_end]
        
        # 预测数据：后future_hours小时
        fut_start = i
        fut_end = i + future_hours
        fut_data = df.iloc[fut_start:fut_end]
        
        if no_hist_power:
            # 无历史发电量模式：只使用预测天气数据
            if fcst_feats:
                X_fcst.append(fut_data[fcst_feats].values)
            
            y.append(fut_data[TARGET_COL].values)
            hours.append(fut_data['Hour'].values)
            dates.append(fut_data['Datetime'].iloc[-1])
            
            # 对于无历史发电量模式，创建空的历史数据数组
            # 确保形状与其他样本一致：(past_hours, n_features)
            if past_hours > 0 and len(hist_feats) > 0:
                empty_hist = np.zeros((past_hours, len(hist_feats)))
            elif past_hours == 0 and len(hist_feats) > 0:
                # 当past_hours=0时，创建形状为(1, n_features)的数组
                empty_hist = np.zeros((1, len(hist_feats)))
            elif past_hours > 0 and len(hist_feats) == 0:
                # 当n_features=0时，创建形状为(past_hours, 0)的数组
                empty_hist = np.zeros((past_hours, 0))
            else:
                # 完全空的情况，创建形状为(1, 0)的数组
                empty_hist = np.zeros((1, 0))
            X_hist.append(empty_hist)
        else:
            # 正常模式：使用历史数据
            # 检查历史数据是否完整
            if len(hist_data) < past_hours:
                continue
            
            # 检查预测数据是否完整
            if len(fut_data) < future_hours:
                continue
            
            # 检查历史数据中是否有NaN值（由于shift操作产生）
            if hist_data[hist_feats].isnull().any().any():
                continue  # 跳过包含NaN的样本
            
            # 构建样本：历史数据形状为 (past_hours, n_features)
            # 这样可以根据past_hours动态调整序列长度
            X_hist.append(hist_data[hist_feats].values)
            
            if fcst_feats:
                # 预测天气：使用预测时间段的天气数据
                X_fcst.append(fut_data[fcst_feats].values)
            
            y.append(fut_data[TARGET_COL].values)
            hours.append(fut_data['Hour'].values)
            dates.append(fut_data['Datetime'].iloc[-1])
    
    if len(X_hist) == 0:
        raise ValueError("无法创建任何有效样本")
    
    # 转换为numpy数组
    X_hist = np.array(X_hist)
    y = np.array(y)
    
    if X_fcst is not None:
        X_fcst = np.array(X_fcst)
    
    return X_hist, X_fcst, y, hours, dates

def split_data(X_hist, X_fcst, y, hours, dates, train_ratio=0.8, val_ratio=0.1, shuffle=True, random_state=42):
    """
    分割数据为训练集、验证集和测试集
    由于样本已经是非连续的时间窗口，可以安全地shuffle和按比例分割
    每个样本都是独立的预测日，不存在数据泄漏问题
    """
    N = X_hist.shape[0]
    
    # 创建随机索引
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(N)
    else:
        indices = np.arange(N)
    
    # 计算分割点
    i_tr = int(N * train_ratio)
    i_val = int(N * (train_ratio + val_ratio))
    
    # 分割索引
    train_idx = indices[:i_tr]
    val_idx = indices[i_tr:i_val]
    test_idx = indices[i_val:]
    
    # 定义切片函数
    def slice_array(arr, indices):
        if isinstance(arr, np.ndarray):
            return arr[indices]
        else:
            # 处理列表类型
            return [arr[i] for i in indices]

    # 分割所有数组
    Xh_tr, Xh_va, Xh_te = slice_array(X_hist, train_idx), slice_array(X_hist, val_idx), slice_array(X_hist, test_idx)
    y_tr, y_va, y_te = slice_array(y, train_idx), slice_array(y, val_idx), slice_array(y, test_idx)
    hrs_tr, hrs_va, hrs_te = slice_array(hours, train_idx), slice_array(hours, val_idx), slice_array(hours, test_idx)
    
    # 处理日期列表
    dates_tr = [dates[i] for i in train_idx]
    dates_va = [dates[i] for i in val_idx]
    dates_te = [dates[i] for i in test_idx]

    # 处理预测特征
    if X_fcst is not None:
        Xf_tr, Xf_va, Xf_te = slice_array(X_fcst, train_idx), slice_array(X_fcst, val_idx), slice_array(X_fcst, test_idx)
    else:
        Xf_tr = Xf_va = Xf_te = None

    return (
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
        Xh_va, Xf_va, y_va, hrs_va, dates_va,
        Xh_te, Xf_te, y_te, hrs_te, dates_te
    )
