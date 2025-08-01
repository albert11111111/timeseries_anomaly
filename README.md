# 时间序列异常检测项目

## 项目简介

本项目使用深度学习模型（TimesNet、Transformer、Autoformer）对灯杆运行数据进行异常检测，并生成日级别的异常标签。项目实现了完整的数据处理、模型训练、结果集成和评估流程。

## 项目特点

- 🚀 **多模型集成**: 使用TimesNet、Transformer、Autoformer三个先进的时间序列模型
- 📊 **智能数据处理**: 自动将Excel数据转换为标准SMAP格式
- 🎯 **精确异常检测**: 基于滑动窗口和动态阈值的异常检测算法
- 📈 **完整评估体系**: 支持误报、漏报、准确率、召回率等多维度评估
- 🔄 **自动化流程**: 一键运行，从数据处理到结果生成的完整自动化

## 环境要求

- Python 3.8+
- 内存: 至少8GB RAM
- 存储: 至少2GB可用空间
- 操作系统: Windows/Linux/macOS

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd 时间序列
```

### 2. 创建虚拟环境

推荐使用conda创建虚拟环境：

```bash
# 创建conda环境
conda create -n timeseries_anomaly python=3.8 -y
conda activate timeseries_anomaly

# 或者使用venv
python -m venv timeseries_env
# Windows
timeseries_env\Scripts\activate
# Linux/macOS
source timeseries_env/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备数据

确保以下数据文件存在于项目根目录：
- `Light_data.xlsx`: 原始灯杆数据（29,892条记录，36个特征）
- `score.xlsx`: 评分文件（包含真实标签）

### 5. 运行项目

```bash
# 步骤1: 数据处理
python anomaly_detection_solution.py

# 步骤2: 模型训练（自动检查并运行未完成的模型）
python check_and_run_models.py

# 步骤3: 结果生成
python anomaly_detection_solution.py

#(多模型结果)
python correct_all_models.py
```

## 项目结构

```
时间序列/
├── README.md                           # 项目说明文档
├── requirements.txt                    # 依赖包列表
├── Light_data.xlsx                     # 原始灯杆数据
├── score.xlsx                          # 评分文件
├── anomaly_detection_solution.py       # 数据处理脚本
├── check_and_run_models.py            # 模型检查和运行脚本
├── process_results.py                  # 结果处理脚本
├── dataset/LIGHT_SMAP/                 # 转换后的数据
│   ├── LIGHT_SMAP_train.npy
│   ├── LIGHT_SMAP_test.npy
│   └── LIGHT_SMAP_test_label.npy
└── Time-Series-Library/               # 模型库
    ├── dataset/LIGHT_SMAP/            # 模型训练数据
    ├── results/                       # 模型预测结果
    └── scripts/anomaly_detection/LIGHT_SMAP/  # 训练脚本
```

## 核心功能

### 数据处理 (`anomaly_detection_solution.py`)

- 加载并预处理Excel数据
- 特征选择和标准化
- 按时间分割训练/测试数据
- 生成SMAP标准格式数据
- 创建模型训练脚本

### 模型训练 (`check_and_run_models.py`)

- 自动检查已完成的模型
- 并行训练多个深度学习模型
- 支持TimesNet、Transformer、Autoformer
- 生成异常检测分数

### 结果处理 (`anomaly_detection_solution.py`)

- 集成多个模型的预测结果
- 点级别到日级别的标签转换
- 动态阈值异常检测
- 多数投票集成策略
- 生成最终评估结果

## 模型说明

### TimesNet
- 基于时频域分析的时间序列模型
- 适合处理多变量时间序列异常检测
- 能够捕获复杂的周期性模式

### Transformer
- 基于注意力机制的经典模型
- 能够捕获长期依赖关系
- 在时间序列任务中表现优异

### Autoformer
- 改进的Transformer模型
- 具有自动相关机制
- 专门为时间序列预测优化

## 技术参数

- **序列长度**: 100个时间步
- **特征数量**: 25个回路特征
- **异常阈值**: 第95百分位数
- **训练轮数**: 10轮
- **批处理大小**: 128
- **模型维度**: 128

## 输出结果

### 主要输出文件

1. **`score_with_predictions.xlsx`**
   - 日期: 2022-07-01 到 2022-12-31
   - 真实标签: 原始真实异常标签
   - 算法标签: 模型预测的异常标签
   - 最终评分: 基于误报和漏报的扣分

2. **模型预测结果**
   - 存储在`Time-Series-Library/results/`目录
   - 包含每个模型的异常检测分数

### 评估指标

- **误报率**: 错误预测为异常的比例
- **漏报率**: 未能检测到真实异常的比例
- **准确率**: 正确预测的比例
- **召回率**: 检测到的真实异常比例
- **F1分数**: 准确率和召回率的调和平均

## 评分规则

- 误报: 扣1分
- 漏报: 扣20分
- 目标: 总评分越接近0越好

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少批处理大小
   # 修改check_and_run_models.py中的batch_size参数
   ```

2. **训练时间长**
   ```bash
   # 减少训练轮数
   # 修改train_epochs参数
   ```

3. **依赖包冲突**
   ```bash
   # 重新创建虚拟环境
   conda create -n timeseries_anomaly python=3.8 -y
   conda activate timeseries_anomaly
   pip install -r requirements.txt
   ```

4. **数据文件不存在**
   ```bash
   # 确保Light_data.xlsx和score.xlsx在项目根目录
   ls -la *.xlsx
   ```

### 参数调优

- `threshold_percentile`: 异常阈值百分位数 (推荐: 90-98)
- `train_epochs`: 训练轮数 (推荐: 5-15)
- `seq_len`: 序列长度 (推荐: 50-200)
- `batch_size`: 批处理大小 (推荐: 64-256)

## 开发指南

### 添加新模型

1. 在`Time-Series-Library/models/`中添加模型文件
2. 在`anomaly_detection_solution.py`中创建训练脚本
3. 在`check_and_run_models.py`中添加模型检查逻辑

### 自定义数据处理

1. 修改`anomaly_detection_solution.py`中的特征选择逻辑
2. 调整数据标准化方法
3. 修改训练/测试数据分割策略

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request


## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至项目维护者

## 致谢

- 感谢Time-Series-Library提供的模型实现
- 感谢所有贡献者的支持

---

**注意**: 首次运行可能需要较长时间进行模型训练，请耐心等待。建议在GPU环境下运行以获得更好的性能。 