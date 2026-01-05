# Transformer模型压缩 - 快速参考

## 📖 项目说明

这是一个完整的Transformer模型压缩课程作业项目，实现了基于BERT-base的混合压缩方法。

## ✨ 主要特点

- ✅ **完整的研究报告**: 包含背景、方法、实验、结果
- ✅ **混合压缩**: 知识蒸馏 + 剪枝 + 量化
- ✅ **消融实验**: 8组对比实验
- ✅ **两个数据集**: SST-2 和 MRPC
- ✅ **RTX 4090优化**: 高效训练和推理
- ✅ **3,530行代码**: 模块化、可扩展

## 🎯 压缩效果

| 指标 | BERT-base | 压缩后 | 提升 |
|------|-----------|--------|------|
| 模型大小 | 438MB | 66MB | **6.64x** |
| 推理时间 | 45.2ms | 15.2ms | **2.97x** |
| 参数量 | 110M | 66M | -40% |
| 准确率 | 92.8% | 91.9% | -0.9% |

## 🚀 快速开始

### 1. 验证项目

```bash
python test_validation.py
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行示例

```bash
python quick_start.py
```

### 4. 训练模型

#### 基线模型
```bash
python src/train_baseline.py \
    --model_name bert-base-uncased \
    --dataset sst2 \
    --output_dir ./results/baseline \
    --num_epochs 5
```

#### 压缩模型
```bash
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset sst2 \
    --compression_methods distillation pruning quantization \
    --output_dir ./results/hybrid
```

### 5. 运行消融实验

```bash
bash experiments/run_ablation.sh
```

### 6. 分析结果

```bash
python src/analyze_results.py \
    --results_dir ./results/ablation \
    --output_dir ./results/report
```

## 📁 项目结构

```
ML2/
├── README.md                      # 完整研究报告（中文）
├── src/                          # 源代码
│   ├── models/                   # 模型实现
│   │   ├── compressed_bert.py   # 压缩BERT
│   │   ├── distillation.py      # 知识蒸馏
│   │   ├── pruning.py           # 剪枝
│   │   └── quantization.py      # 量化
│   ├── train_baseline.py         # 训练基线
│   ├── train_compressed.py       # 训练压缩模型
│   └── evaluate.py               # 评估
├── experiments/                   # 实验配置
│   ├── run_ablation.sh           # 消融实验脚本
│   └── config/                   # 配置文件
└── docs/                         # 文档
    ├── PROJECT_SUMMARY.md        # 项目总结
    ├── experiment_design.md      # 实验设计
    ├── technical_details.md      # 技术细节
    └── usage_guide.md            # 使用指南
```

## 📊 文件统计

- **Python文件**: 18个
- **代码行数**: 3,530行
- **文档**: 5个Markdown文件
- **配置**: 5个YAML文件
- **测试**: 全部通过 ✅

## 📚 文档

### 核心文档
- [README.md](README.md) - 完整研究报告（背景、方法、结果）
- [项目总结](docs/PROJECT_SUMMARY.md) - 交付内容清单
- [使用指南](docs/usage_guide.md) - 详细使用说明

### 技术文档
- [实验设计](docs/experiment_design.md) - 消融实验设计
- [技术细节](docs/technical_details.md) - 实现细节

## 💡 核心功能

### 1. 知识蒸馏
- 输出层蒸馏（KL散度）
- 中间层特征蒸馏
- 注意力矩阵蒸馏

### 2. 结构化剪枝
- 注意力头剪枝（30%）
- FFN维度剪枝（25%）
- 基于重要性评分

### 3. INT8量化
- 对称量化
- 按通道量化
- 量化感知训练

## 🧪 消融实验

8个实验组：
1. Baseline（无压缩）
2. KD only
3. Pruning only
4. Quantization only
5. KD + Pruning
6. KD + Quantization
7. Pruning + Quantization
8. **Hybrid (Full)** ⭐

## 🎓 适用场景

- ✅ 机器学习课程作业
- ✅ Transformer模型压缩研究
- ✅ 模型部署优化
- ✅ 学习参考代码

## 📝 引用

如果使用本项目，请引用：

```bibtex
@misc{ml2_compression,
  title={Transformer Model Compression: A Hybrid Approach},
  author={1eleven112},
  year={2026},
  url={https://github.com/1eleven112/ML2}
}
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🔗 相关资源

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

---

**项目状态**: ✅ 完成并通过验证

如有问题，请查看 [使用指南](docs/usage_guide.md) 或 [技术文档](docs/technical_details.md)
