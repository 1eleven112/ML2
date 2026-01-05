# 使用指南

## 快速开始

### 1. 环境准备

#### 安装依赖

```bash
pip install -r requirements.txt
```

如果在RTX 4090上运行，确保安装了CUDA 11.8+和对应的PyTorch版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 验证安装

```bash
python test_validation.py
```

### 2. 快速示例

运行快速开始示例以验证环境：

```bash
python quick_start.py
```

## 训练模型

### 训练基线BERT模型

在SST-2数据集上训练标准BERT-base模型：

```bash
python src/train_baseline.py \
    --model_name bert-base-uncased \
    --dataset sst2 \
    --output_dir ./results/baseline_sst2 \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5
```

在MRPC数据集上训练：

```bash
python src/train_baseline.py \
    --model_name bert-base-uncased \
    --dataset mrpc \
    --output_dir ./results/baseline_mrpc \
    --num_epochs 5 \
    --batch_size 32
```

### 训练压缩模型

#### 仅知识蒸馏

```bash
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset sst2 \
    --compression_methods distillation \
    --output_dir ./results/kd_only \
    --num_epochs_kd 20 \
    --batch_size 32
```

#### 仅剪枝

```bash
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset sst2 \
    --compression_methods pruning \
    --head_pruning_ratio 0.3 \
    --ffn_pruning_ratio 0.25 \
    --output_dir ./results/pruning_only \
    --num_epochs_pruning 10 \
    --batch_size 32
```

#### 仅量化

```bash
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset sst2 \
    --compression_methods quantization \
    --quantization_bits 8 \
    --output_dir ./results/quantization_only \
    --num_epochs_quantization 5 \
    --batch_size 32
```

#### 完整混合压缩

```bash
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset sst2 \
    --compression_methods distillation pruning quantization \
    --head_pruning_ratio 0.3 \
    --ffn_pruning_ratio 0.25 \
    --quantization_bits 8 \
    --output_dir ./results/hybrid_full \
    --num_epochs_kd 20 \
    --num_epochs_pruning 10 \
    --num_epochs_quantization 5 \
    --batch_size 32 \
    --learning_rate 2e-5
```

### 调整超参数

根据显存大小调整批次大小：

```bash
# RTX 4090 (24GB) - 可以使用较大批次
--batch_size 64

# RTX 3090 (24GB)
--batch_size 32

# RTX 3080 (10GB)
--batch_size 16

# 显存不足时使用梯度累积
--batch_size 8 --gradient_accumulation_steps 4
```

## 评估模型

### 评估单个模型

```bash
python src/evaluate.py \
    --model_path ./results/hybrid_full/final_model \
    --dataset sst2 \
    --batch_size 32
```

### 评估所有模型

```bash
for model_dir in ./results/*/final_model; do
    if [ -d "$model_dir" ]; then
        python src/evaluate.py \
            --model_path $model_dir \
            --dataset sst2 \
            --batch_size 32
    fi
done
```

## 运行消融实验

### 运行完整消融实验

```bash
bash experiments/run_ablation.sh
```

这会运行所有8个实验组：
1. Baseline
2. KD only
3. Pruning only
4. Quantization only
5. KD + Pruning
6. KD + Quantization
7. Pruning + Quantization
8. Hybrid (Full)

### 运行部分实验

编辑 `experiments/run_ablation.sh` 并注释掉不需要的实验。

### 自定义实验配置

修改 `experiments/config/` 目录下的YAML配置文件。

## 结果分析

### 生成分析报告

```bash
python src/analyze_results.py \
    --results_dir ./results/ablation \
    --dataset sst2 \
    --output_dir ./results/ablation/report
```

这会生成：
- `comparison_table.csv`: 对比表格
- `ablation_results.png`: 消融实验可视化
- `compression_tradeoff.png`: 压缩-性能权衡图
- `results_table.tex`: LaTeX格式表格
- `REPORT.md`: Markdown格式报告

### 查看训练历史

训练历史保存在 `training_history.json` 文件中：

```bash
cat ./results/baseline/training_history.json
```

## 常见问题

### Q1: CUDA out of memory

**解决方案**：
- 减小批次大小：`--batch_size 16`
- 减小最大序列长度：`--max_length 64`
- 使用梯度累积
- 在CPU上训练（较慢）

### Q2: 下载数据集失败

**解决方案**：
- 使用镜像源
- 手动下载数据集并放置在 `.cache/huggingface/datasets/` 目录
- 设置代理：`export HF_ENDPOINT=https://hf-mirror.com`

### Q3: 训练时间过长

**解决方案**：
- 减少训练轮数
- 使用更小的数据集（如CoLA）
- 使用预训练的压缩模型作为起点
- 启用混合精度训练

### Q4: 模型性能不佳

**解决方案**：
- 增加训练轮数
- 调整学习率（尝试2e-5, 3e-5, 5e-5）
- 降低压缩比例（减少剪枝率）
- 增加蒸馏训练轮数
- 使用预热策略

### Q5: 找不到保存的模型

**解决方案**：
检查输出目录，模型保存在：
- `{output_dir}/best_model/` - 最佳模型
- `{output_dir}/final_model/` - 最终模型
- `{output_dir}/kd_model/` - 蒸馏后的模型（如果使用）
- `{output_dir}/pruned_model/` - 剪枝后的模型（如果使用）

## 高级用法

### 使用配置文件

创建自定义配置文件 `my_config.yaml`：

```yaml
model:
  base_model: bert-base-uncased
  num_labels: 2

compression:
  methods: [distillation, pruning, quantization]
  pruning_ratio: 0.5
  quantization_bits: 8

training:
  num_epochs_kd: 15
  num_epochs_pruning: 8
  num_epochs_quantization: 5
  batch_size: 32
  learning_rate: 2.0e-5
```

### 自定义数据集

修改 `src/data/dataset_loader.py` 以支持自定义数据集。

### 导出模型

将模型导出为ONNX格式：

```python
import torch
from transformers import AutoTokenizer

model = CompressedBertForSequenceClassification.load_compressed_model(
    "./results/hybrid_full/final_model"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dummy_input = tokenizer("This is a test", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "compressed_bert.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'}
    }
)
```

### 在推理中使用

```python
import torch
from transformers import AutoTokenizer
from src.models.compressed_bert import CompressedBertForSequenceClassification

# 加载模型和tokenizer
model = CompressedBertForSequenceClassification.load_compressed_model(
    "./results/hybrid_full/final_model"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 推理
text = "This movie is fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs['logits'], dim=-1)
    print(f"Prediction: {prediction.item()}")  # 0=Negative, 1=Positive
```

## 性能优化建议

### 训练优化

1. **使用混合精度训练**（FP16）可以加速约2倍
2. **预先缓存数据集**避免重复加载
3. **使用多GPU并行训练**（如果有多张GPU）
4. **适当增加批次大小**以充分利用GPU

### 推理优化

1. **批处理推理**而不是单条推理
2. **使用TorchScript**进行JIT编译
3. **量化到INT8**以加速推理
4. **使用ONNX Runtime**或TensorRT

### 内存优化

1. **梯度检查点**（gradient checkpointing）
2. **及时清理缓存**：`torch.cuda.empty_cache()`
3. **使用更小的数据类型**（FP16而不是FP32）
4. **减小序列长度**和批次大小

## 实验建议

### 研究方向

1. **更激进的压缩**：
   - 剪枝率提高到60%+
   - 4-bit或2-bit量化
   - 层级剪枝（减少Transformer层数）

2. **其他压缩技术**：
   - 权重共享
   - 低秩分解
   - 神经架构搜索

3. **应用场景**：
   - 移动端部署
   - 边缘设备推理
   - 实时应用

4. **其他模型**：
   - BERT-large
   - RoBERTa
   - DistilBERT

### 论文写作

使用生成的结果和图表撰写论文：

1. 引言：参考 `README.md` 的背景部分
2. 方法：参考 `docs/technical_details.md`
3. 实验：使用 `docs/experiment_design.md` 的设置
4. 结果：使用 `results/ablation/report/` 中的图表
5. 结论：总结关键发现和未来工作

## 资源

### 文档

- [README.md](../README.md) - 项目概述和结果
- [实验设计](../docs/experiment_design.md) - 详细的实验设计
- [技术细节](../docs/technical_details.md) - 实现细节

### 代码

- `src/models/` - 模型实现
- `src/train_*.py` - 训练脚本
- `src/evaluate.py` - 评估脚本
- `src/analyze_results.py` - 结果分析

### 配置

- `experiments/config/` - 实验配置
- `experiments/run_ablation.sh` - 消融实验脚本

## 技术支持

如果遇到问题：

1. 查看本文档的"常见问题"部分
2. 检查代码中的注释和文档字符串
3. 查看 `docs/` 目录下的详细文档
4. 运行 `python test_validation.py` 验证环境

## 许可证

本项目使用MIT许可证。详见LICENSE文件。
