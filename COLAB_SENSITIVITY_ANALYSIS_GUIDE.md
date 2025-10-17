# Google Colab 敏感性分析运行指南

本指南帮助您在Google Colab上一键运行所有8个敏感性分析实验。

## 🚀 快速开始

### 步骤1：打开Colab并克隆项目

在Colab新建notebook，运行以下代码：

```python
# 克隆项目
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# 安装依赖
!pip install -r requirements.txt
```

### 步骤2：上传数据集

#### 方法A：从本地上传

```python
# 创建data目录
!mkdir -p data

# 上传CSV文件
from google.colab import files
import shutil

uploaded = files.upload()  # 选择您的100个CSV文件

# 移动到data目录
for filename in uploaded.keys():
    shutil.move(filename, f'data/{filename}')
    
print(f"已上传 {len(uploaded)} 个文件")
```

#### 方法B：从Google Drive复制

```python
from google.colab import drive
drive.mount('/content/drive')

# 复制数据文件
!cp /content/drive/MyDrive/your_data_folder/*.csv data/

# 检查文件数量
!ls data/*.csv | wc -l
```

### 步骤3：运行所有实验（一键运行）

```python
# 一键运行所有8个实验
# 结果自动保存到Drive: /content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results
!python run_sensitivity_analysis_colab.py
```

**就这么简单！** 🎉

---

## 📋 高级用法

### 运行指定的实验

```python
# 只运行实验1、3、5
!python run_sensitivity_analysis_colab.py --experiments 1 3 5

# 实验编号对照表：
# 1 = 季节效果分析
# 2 = 小时效果分析
# 3 = 天气特征分层分析
# 4 = 回看窗口长度分析
# 5 = 模型复杂度分析
# 6 = 训练数据规模分析
# 7 = 非shuffle训练分析
# 8 = 数据扩展分析
```

### 自定义输出路径

```python
# 指定自定义输出路径
!python run_sensitivity_analysis_colab.py \
    --custom-output "/content/drive/MyDrive/my_custom_path/results"
```

### 使用自定义数据目录

```python
# 如果数据在其他目录
!python run_sensitivity_analysis_colab.py \
    --data-dir "/content/drive/MyDrive/my_data_folder"
```

### 跳过Drive挂载（如果已经挂载）

```python
# 如果Drive已经挂载，可以跳过挂载步骤
!python run_sensitivity_analysis_colab.py --skip-mount
```

---

## 🎯 8个实验说明

| # | 实验名称 | 分析内容 | 模型 | 预计时间* |
|---|---------|---------|------|----------|
| 1 | 季节效果 | 春夏秋冬4季 | 7+LSR | 2-3小时 |
| 2 | 小时效果 | 24小时 | 7+LSR | 2-3小时 |
| 3 | 天气特征 | 4个特征层级 | 7+LSR | 3-4小时 |
| 4 | 回看窗口 | 4个窗口长度 | 7 | 2-3小时 |
| 5 | 模型复杂度 | 4个复杂度档次 | 7 | 3-4小时 |
| 6 | 训练规模 | 4个训练比例 | 7+LSR | 3-4小时 |
| 7 | 非shuffle | 顺序分割 | 7+LSR | 2-3小时 |
| 8 | 数据扩展 | 小时滑窗 | 7+LSR | 4-6小时 |

\* 预计时间基于100个电站、使用Colab GPU

**总计：约20-30小时**（如果一次运行所有实验）

---

## 📊 查看结果

### 方法1：在Colab中查看

```python
import pandas as pd

# 读取某个实验的聚合结果
df = pd.read_csv('/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results/seasonal_effect_aggregated.csv')
print(df)

# 查看所有生成的结果文件
!ls -lh "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results/"
```

### 方法2：在Google Drive中查看

打开Google Drive，导航到：
```
我的云端硬盘 > Solar PV electricity > sensitivity_analysis_results
```

### 结果文件结构

每个实验生成3个CSV文件：

```
sensitivity_analysis_results/
├── seasonal_effect_detailed.csv        # 详细数据（每个电站）
├── seasonal_effect_aggregated.csv      # 聚合数据（均值+标准差）
├── seasonal_effect_pivot.csv           # 数据透视表
├── hourly_effect_detailed.csv
├── hourly_effect_aggregated.csv
├── hourly_effect_pivot.csv
└── ... (其他实验结果)
```

---

## ⚙️ 配置GPU加速

**强烈建议启用GPU以加速训练！**

1. 点击Colab菜单：`Runtime` > `Change runtime type`
2. 选择 `Hardware accelerator`: **GPU**
3. 点击 `Save`

验证GPU：

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🐛 故障排除

### 问题1：Drive挂载失败

```python
# 手动挂载Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 问题2：找不到CSV文件

```python
# 检查data目录
!ls -la data/

# 检查CSV文件
!ls data/*.csv | head -5
```

### 问题3：内存不足

```python
# 检查内存使用
!free -h

# 清理缓存
!rm -rf __pycache__
!rm -rf */__pycache__

# 重启runtime
# Runtime > Restart runtime
```

### 问题4：实验超时

```python
# 单独运行超时的实验
!python sensitivity_analysis/seasonal_effect.py \
    --data-dir data \
    --output-dir "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results"
```

### 问题5：查看详细错误日志

如果某个实验失败，可以单独运行并查看详细输出：

```python
# 单独运行某个实验
!python sensitivity_analysis/seasonal_effect.py \
    --data-dir data \
    --output-dir "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results"
```

---

## 📝 注意事项

### 1. Colab会话时间限制
- **免费版**：最长12小时，闲置90分钟断开
- **Pro版**：最长24小时，闲置更长时间
- **建议**：分批运行实验，避免一次运行太久

### 2. 分批运行策略

**第一批（较快的实验）**：
```python
!python run_sensitivity_analysis_colab.py --experiments 1 2 7
```

**第二批（中等时长）**：
```python
!python run_sensitivity_analysis_colab.py --experiments 3 4 6
```

**第三批（较慢的实验）**：
```python
!python run_sensitivity_analysis_colab.py --experiments 5 8
```

### 3. 数据要求

确保每个CSV文件包含：
- 时间列：`Year`, `Month`, `Day`, `Hour`
- 目标列：`Capacity Factor`
- 11个气象变量（真实值 + `_pred`后缀）

### 4. 保持连接

在Colab运行长时间任务时：
- 保持浏览器标签页打开
- 可以使用以下代码防止断开：

```python
# 防止Colab断开连接
from IPython.display import Javascript
display(Javascript('''
    function ClickConnect(){
        console.log("Keeping Colab alive");
        document.querySelector("colab-toolbar-button#connect").click()
    }
    setInterval(ClickConnect, 60000)
'''))
```

---

## 📞 获取帮助

如果遇到问题：

1. **查看日志**：脚本会显示详细的错误信息
2. **检查数据**：确保CSV格式正确
3. **查看文档**：参考 `sensitivity_analysis/README.md`
4. **提交Issue**：在GitHub上提交问题

---

## ✅ 完整工作流程示例

```python
# ========================================
# 完整的Colab运行流程
# ========================================

# 1. 克隆项目
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 从Drive复制数据
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/PV_Data/*.csv data/

# 4. 检查环境
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
!ls data/*.csv | wc -l

# 5. 运行所有实验
!python run_sensitivity_analysis_colab.py

# 6. 查看结果
!ls -lh "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results/"

# 7. 读取结果示例
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results/seasonal_effect_aggregated.csv')
print(df)
```

---

## 🎉 就是这样！

运行`run_sensitivity_analysis_colab.py`，坐等结果完成！☕

所有结果会自动保存到您的Google Drive中。

