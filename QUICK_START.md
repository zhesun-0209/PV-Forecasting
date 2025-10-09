# 🚀 快速启动指南 - 100个电站批量实验

## 📝 简明步骤

### 1. 克隆项目
```bash
git clone https://github.com/zhesun-0209/PV-Forecasting.git
cd PV-Forecasting
pip install -r requirements.txt
```

### 2. 准备数据（100个CSV文件）
```bash
# 将所有数据集放到data目录
cp your_datasets/*.csv data/

# 或从Google Drive复制（Colab）
# cp /content/drive/MyDrive/your_folder/*.csv data/
```

### 3. 批量生成配置文件
```bash
python batch_create_configs.py
```

这会自动:
- 扫描 `data/` 目录下的所有CSV
- 提取电站ID
- 检测数据时间范围
- 生成配置文件到 `config/plants/Plant{ID}.yaml`

### 4. 运行实验

**选项A: 全部运行（不推荐，太长）**
```bash
python run_experiments_multi_plant.py
```

**选项B: 分批运行（推荐）**
```bash
# 第1批：前25个电站
python run_experiments_multi_plant.py --max-plants 25

# 第2批：第26-50个
python run_experiments_multi_plant.py --skip 25 --max-plants 25

# 第3批：第51-75个
python run_experiments_multi_plant.py --skip 50 --max-plants 25

# 第4批：第76-100个
python run_experiments_multi_plant.py --skip 75 --max-plants 25
```

**选项C: 指定特定电站**
```bash
python run_experiments_multi_plant.py --plants 1001 1002 1003
```

### 5. 监控进度
```bash
# 方式1：检查进度
python check_progress.py

# 方式2：实时监控
python monitor_progress.py
```

---

## 📊 数据集格式要求

CSV文件必须包含以下列：
- `Year`, `Month`, `Day`, `Hour` - 时间信息
- 功率列（会自动识别）
- 天气特征列（可选）

文件命名建议：
- `Project1001.csv`
- `Plant1001.csv`  
- 或直接 `1001.csv`

---

## ⏱️ 时间估算

- **单个电站**: 284个实验，约2-3小时（GPU）
- **100个电站**: 约200-300小时
- **建议**: 使用4-5个Colab notebook并行运行，每个跑20-25个电站

---

## 💾 结果文件

运行完成后会生成：
- `results_{plant_id}.csv` - 各电站的284个实验结果
- `results/` - 详细输出目录
- 可用 `pandas` 合并分析所有结果

---

## 🔧 Colab专用命令

```python
# 1. 检查GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# 2. 挂载Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. 保存结果到Drive
!cp -r results/ /content/drive/MyDrive/PV_Results/
!cp *.csv /content/drive/MyDrive/PV_Results/

# 4. 下载结果
from google.colab import files
!zip -r results.zip results/ *.csv
files.download('results.zip')
```

---

## ⚠️ 常见问题

### Q: Colab超时了怎么办？
**A**: 脚本支持断点续传，重新运行即可从上次停止的地方继续。

### Q: 内存不足？
**A**: 每完成一个电站就会自动清理内存。如果还是不够，减少 `--max-plants` 参数。

### Q: 如何并行加速？
**A**: 开多个Colab notebook，每个运行不同批次的电站。

---

## 📁 目录结构

```
PV-Forecasting/
├── data/                      # 放置100个CSV文件
│   ├── Project1001.csv
│   ├── Project1002.csv
│   └── ...
├── config/
│   ├── plant_template.yaml    # 配置模板
│   └── plants/                # 自动生成的配置
│       ├── Plant1001.yaml
│       ├── Plant1002.yaml
│       └── ...
├── results/                   # 实验结果
├── batch_create_configs.py    # 批量配置生成脚本
├── run_all_experiments.py     # 单电站实验
└── run_experiments_multi_plant.py  # 多电站批量实验
```

---

## 🎯 完整示例（Colab）

```python
# 步骤1: 克隆和安装
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting
!pip install -q -r requirements.txt

# 步骤2: 从Drive复制数据
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/PV_Datasets/*.csv data/

# 步骤3: 生成配置
!python batch_create_configs.py

# 步骤4: 运行第1批（前25个电站）
!python run_experiments_multi_plant.py --max-plants 25

# 步骤5: 保存结果
!cp -r results/ /content/drive/MyDrive/PV_Results_Batch1/
!cp *.csv /content/drive/MyDrive/PV_Results_Batch1/

print("✓ 第1批完成！")
```

---

## 📞 需要帮助？

查看完整文档：
- `COLAB_SETUP_GUIDE.md` - 详细Colab指南
- `README.md` - 项目说明
- GitHub Issues - 提交问题

---

**祝实验顺利！**  
如有问题，随时查看文档或提Issue。

