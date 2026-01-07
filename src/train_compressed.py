"""
训练压缩BERT模型（混合压缩方法）
"""

import argparse
import os
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.compressed_bert import CompressedBertForSequenceClassification, CompressedBertConfig
from models.distillation import create_distillation_trainer
from models.pruning import BertPruner
from models.quantization import quantize_model, QuantizationConfig
from data.dataset_loader import load_glue_data
from utils.metrics import evaluate_model


def train_compressed_model(args):
    """训练压缩模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, num_labels = load_glue_data(
        dataset_name=args.dataset,
        model_name=args.teacher_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    
    # 创建学生模型
    print("Creating student model...")
    config = CompressedBertConfig(
        base_model_name=args.teacher_model,
        pruning_ratio=args.pruning_ratio,
        head_pruning_ratio=args.head_pruning_ratio,
        ffn_pruning_ratio=args.ffn_pruning_ratio,
        quantization_bits=args.quantization_bits,
        use_distillation='distillation' in args.compression_methods,
    )
    
    student_model = CompressedBertForSequenceClassification(
        config=config,
        num_labels=num_labels,
        from_pretrained=True
    )
    student_model.to(device)
    
    print(f"Original model size: {student_model.get_model_size():.2f} MB")
    print(f"Original parameters: {student_model.get_num_parameters() / 1e6:.2f}M")
    
    # 阶段1: 知识蒸馏
    if 'distillation' in args.compression_methods:
        print("\n" + "="*50)
        print("Phase 1: Knowledge Distillation")
        print("="*50)
        
        distillation_trainer = create_distillation_trainer(
            teacher_model_name=args.teacher_model,
            student_model=student_model,
            device=device,
        )
        
        optimizer = AdamW(
            student_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        num_epochs_kd = args.num_epochs_kd
        total_steps = len(train_loader) * num_epochs_kd
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        best_val_metric = 0
        for epoch in range(num_epochs_kd):
            print(f"\nKD Epoch {epoch + 1}/{num_epochs_kd}")
            
            student_model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc="Distillation")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                # 计算蒸馏损失
                loss_dict = distillation_trainer.compute_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    token_type_ids=token_type_ids,
                )
                
                loss = loss_dict['loss']
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 验证
            val_metrics = evaluate_model(student_model, val_loader, device, args.dataset)
            print(f"KD Epoch {epoch + 1} - Val Metrics: {val_metrics}")
            
            # 保存最佳模型
            if val_metrics.get('accuracy', 0) > best_val_metric:
                best_val_metric = val_metrics.get('accuracy', 0)
                os.makedirs(args.output_dir, exist_ok=True)
                student_model.save_compressed_model(
                    os.path.join(args.output_dir, "kd_model")
                )
    
    # 阶段2: 结构化剪枝
    if 'pruning' in args.compression_methods:
        print("\n" + "="*50)
        print("Phase 2: Structured Pruning")
        print("="*50)
        
        pruner = BertPruner(
            model=student_model,
            head_pruning_ratio=args.head_pruning_ratio,
            ffn_pruning_ratio=args.ffn_pruning_ratio,
        )
        
        # 应用剪枝
        pruning_stats = pruner.apply_pruning(train_loader, device)
        
        print(f"\nModel size after pruning: {student_model.get_model_size():.2f} MB")
        print(f"Parameters after pruning: {student_model.get_num_parameters() / 1e6:.2f}M")
        
        # 微调剪枝后的模型
        optimizer = AdamW(
            student_model.parameters(),
            lr=args.learning_rate * 0.5,  # 降低学习率
            weight_decay=args.weight_decay
        )
        
        num_epochs_prune = args.num_epochs_pruning
        total_steps = len(train_loader) * num_epochs_prune
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        for epoch in range(num_epochs_prune):
            print(f"\nPruning Fine-tune Epoch {epoch + 1}/{num_epochs_prune}")
            
            student_model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc="Pruning Fine-tune")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                
                loss = outputs['loss']
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 验证
            val_metrics = evaluate_model(student_model, val_loader, device, args.dataset)
            print(f"Pruning Epoch {epoch + 1} - Val Metrics: {val_metrics}")
        
        # 保存剪枝后的模型
        student_model.save_compressed_model(
            os.path.join(args.output_dir, "pruned_model")
        )
    
    # 阶段3: 量化
    if 'quantization' in args.compression_methods:
        print("\n" + "="*50)
        print("Phase 3: Quantization")
        print("="*50)
        
        quant_config = QuantizationConfig(
            bits=args.quantization_bits,
            quantize_embeddings=False,
            quantize_classifier=True,
        )
        
        # 量化模型
        quantized_model = quantize_model(
            model=student_model,
            config=quant_config,
            dataloader=train_loader,
            device=device,
        )
        
        print(f"Model size after quantization: {quantized_model.get_model_size():.2f} MB")
        
        # 量化感知训练
        optimizer = AdamW(
            quantized_model.parameters(),
            lr=args.learning_rate * 0.1,  # 更低的学习率
            weight_decay=args.weight_decay
        )
        
        num_epochs_quant = args.num_epochs_quantization
        
        for epoch in range(num_epochs_quant):
            print(f"\nQuantization Training Epoch {epoch + 1}/{num_epochs_quant}")
            
            quantized_model.train()
            
            progress_bar = tqdm(train_loader, desc="Quantization Training")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                outputs = quantized_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                
                loss = outputs['loss']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 验证
            val_metrics = evaluate_model(quantized_model, val_loader, device, args.dataset)
            print(f"Quantization Epoch {epoch + 1} - Val Metrics: {val_metrics}")
        
        student_model = quantized_model
    
    # 最终评估
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    val_metrics = evaluate_model(student_model, val_loader, device, args.dataset)
    print(f"Final Validation Metrics: {val_metrics}")
    
    if test_loader is not None:
        test_metrics = evaluate_model(student_model, test_loader, device, args.dataset)
        print(f"Final Test Metrics: {test_metrics}")
    
    # 保存最终模型
    student_model.save_compressed_model(os.path.join(args.output_dir, "final_model"))
    
    # 测量推理时间
    print("\nMeasuring inference time...")
    student_model.eval()
    sample_batch = next(iter(val_loader))
    input_ids = sample_batch['input_ids'].to(device)
    attention_mask = sample_batch['attention_mask'].to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = student_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 测量
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = student_model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    
    print(f"\nCompression completed! Final model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train compressed BERT model")
    
    # 模型参数
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="bert-base-uncased",
        help="Teacher model name"
    )
    
    # 压缩方法
    parser.add_argument(
        "--compression_methods",
        nargs='+',
        default=['distillation', 'pruning', 'quantization'],
        choices=['distillation', 'pruning', 'quantization'],
        help="Compression methods to apply"
    )
    
    # 剪枝参数
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.4,
        help="Overall pruning ratio"
    )
    parser.add_argument(
        "--head_pruning_ratio",
        type=float,
        default=0.3,
        help="Attention head pruning ratio"
    )
    parser.add_argument(
        "--ffn_pruning_ratio",
        type=float,
        default=0.25,
        help="FFN dimension pruning ratio"
    )
    
    # 量化参数
    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=8,
        help="Quantization bits"
    )
    
    # 数据参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "mrpc", "cola", "qnli", "qqp", "rte"],
        help="GLUE dataset name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    # 训练参数
    parser.add_argument(
        "--num_epochs_kd",
        type=int,
        default=20,
        help="Number of knowledge distillation epochs"
    )
    parser.add_argument(
        "--num_epochs_pruning",
        type=int,
        default=10,
        help="Number of pruning fine-tuning epochs"
    )
    parser.add_argument(
        "--num_epochs_quantization",
        type=int,
        default=5,
        help="Number of quantization training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/compressed",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    train_compressed_model(args)


if __name__ == "__main__":
    main()
