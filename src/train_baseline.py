"""
训练基线BERT模型
"""

import argparse
import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_glue_data
from utils.metrics import evaluate_model


def train_baseline(args):
    """训练基线模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, num_labels = load_glue_data(
        dataset_name=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    
    # 加载模型
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    model.to(device)
    
    # 优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_metric = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            token_type_ids = None
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # 统计
            train_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct / train_total:.4f}'
            })
        
        # 计算训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        print("Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device, args.dataset)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_metrics.get('loss', 0))
        history['val_acc'].append(val_metrics.get('accuracy', 0))
        
        # 打印结果
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Metrics: {val_metrics}")
        
        # 保存最佳模型
        primary_metric = val_metrics.get('accuracy', 0)
        if primary_metric > best_val_metric:
            best_val_metric = primary_metric
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"Best model saved with {primary_metric:.4f}")
    
    # 保存最终模型
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # 测试集评估（如果有）
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device, args.dataset)
        print(f"Test Metrics: {test_metrics}")
    
    # 保存训练历史
    import json
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed! Models saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline BERT model")
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name"
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
    
    # 训练参数
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
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
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/baseline",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    train_baseline(args)


if __name__ == "__main__":
    main()
