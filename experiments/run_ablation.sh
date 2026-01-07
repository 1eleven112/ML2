#!/bin/bash
# 消融实验脚本 - 在不同压缩方法组合下训练模型

DATASET="sst2"
BATCH_SIZE=32
OUTPUT_BASE="./results/ablation"

echo "Starting Ablation Experiments for $DATASET dataset"
echo "=================================================="

# 实验1: Baseline (无压缩)
echo ""
echo "Experiment 1: Baseline (No Compression)"
python src/train_baseline.py \
    --model_name bert-base-uncased \
    --dataset $DATASET \
    --output_dir $OUTPUT_BASE/baseline \
    --num_epochs 3 \
    --batch_size $BATCH_SIZE

# 实验2: 仅知识蒸馏
echo ""
echo "Experiment 2: Knowledge Distillation Only"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods distillation \
    --output_dir $OUTPUT_BASE/kd_only \
    --num_epochs_kd 15 \
    --batch_size $BATCH_SIZE

# 实验3: 仅剪枝
echo ""
echo "Experiment 3: Pruning Only"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods pruning \
    --head_pruning_ratio 0.3 \
    --ffn_pruning_ratio 0.25 \
    --output_dir $OUTPUT_BASE/pruning_only \
    --num_epochs_pruning 10 \
    --batch_size $BATCH_SIZE

# 实验4: 仅量化
echo ""
echo "Experiment 4: Quantization Only"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods quantization \
    --quantization_bits 8 \
    --output_dir $OUTPUT_BASE/quantization_only \
    --num_epochs_quantization 5 \
    --batch_size $BATCH_SIZE

# 实验5: 知识蒸馏 + 剪枝
echo ""
echo "Experiment 5: KD + Pruning"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods distillation pruning \
    --head_pruning_ratio 0.3 \
    --ffn_pruning_ratio 0.25 \
    --output_dir $OUTPUT_BASE/kd_pruning \
    --num_epochs_kd 15 \
    --num_epochs_pruning 8 \
    --batch_size $BATCH_SIZE

# 实验6: 知识蒸馏 + 量化
echo ""
echo "Experiment 6: KD + Quantization"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods distillation quantization \
    --quantization_bits 8 \
    --output_dir $OUTPUT_BASE/kd_quantization \
    --num_epochs_kd 15 \
    --num_epochs_quantization 5 \
    --batch_size $BATCH_SIZE

# 实验7: 剪枝 + 量化
echo ""
echo "Experiment 7: Pruning + Quantization"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods pruning quantization \
    --head_pruning_ratio 0.3 \
    --ffn_pruning_ratio 0.25 \
    --quantization_bits 8 \
    --output_dir $OUTPUT_BASE/pruning_quantization \
    --num_epochs_pruning 8 \
    --num_epochs_quantization 5 \
    --batch_size $BATCH_SIZE

# 实验8: 完整混合方法 (KD + Pruning + Quantization)
echo ""
echo "Experiment 8: Full Hybrid (KD + Pruning + Quantization)"
python src/train_compressed.py \
    --teacher_model bert-base-uncased \
    --dataset $DATASET \
    --compression_methods distillation pruning quantization \
    --head_pruning_ratio 0.3 \
    --ffn_pruning_ratio 0.25 \
    --quantization_bits 8 \
    --output_dir $OUTPUT_BASE/hybrid_full \
    --num_epochs_kd 15 \
    --num_epochs_pruning 8 \
    --num_epochs_quantization 5 \
    --batch_size $BATCH_SIZE

echo ""
echo "=================================================="
echo "All ablation experiments completed!"
echo "Results saved in $OUTPUT_BASE/"
echo ""
echo "Evaluating all models..."

# 评估所有模型
for exp_dir in $OUTPUT_BASE/*/; do
    if [ -d "$exp_dir/final_model" ]; then
        echo "Evaluating $(basename $exp_dir)..."
        python src/evaluate.py \
            --model_path $exp_dir/final_model \
            --dataset $DATASET \
            --batch_size $BATCH_SIZE
    fi
done

echo ""
echo "All evaluations completed!"
