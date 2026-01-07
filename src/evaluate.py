"""
评估模型性能
"""

import argparse
import os
import torch
import time
import json
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.compressed_bert import CompressedBertForSequenceClassification
from data.dataset_loader import load_glue_data
from utils.metrics import evaluate_model


def evaluate(args):
    """评估模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading {args.dataset} dataset...")
    _, val_loader, test_loader, num_labels = load_glue_data(
        dataset_name=args.dataset,
        model_name="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    try:
        model = CompressedBertForSequenceClassification.load_compressed_model(args.model_path)
    except:
        # 如果加载失败，尝试使用transformers加载
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    model.to(device)
    model.eval()
    
    # 获取模型信息
    print("\nModel Information:")
    if hasattr(model, 'get_model_size'):
        print(f"  Model size: {model.get_model_size():.2f} MB")
        print(f"  Parameters: {model.get_num_parameters() / 1e6:.2f}M")
    else:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    # 评估
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, args.dataset)
    print(f"Validation Metrics: {val_metrics}")
    
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device, args.dataset)
        print(f"Test Metrics: {test_metrics}")
    else:
        test_metrics = None
    
    # 测量推理时间
    print("\nMeasuring inference time...")
    sample_batch = next(iter(val_loader))
    input_ids = sample_batch['input_ids'].to(device)
    attention_mask = sample_batch['attention_mask'].to(device)
    token_type_ids = sample_batch.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            if token_type_ids is not None:
                _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 测量
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            if token_type_ids is not None:
                _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
    
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    print(f"Throughput: {1000 / avg_inference_time * args.batch_size:.2f} samples/sec")
    
    # 保存结果
    results = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "inference_time_ms": avg_inference_time,
        "throughput": 1000 / avg_inference_time * args.batch_size,
    }
    
    if hasattr(model, 'get_model_size'):
        results["model_size_mb"] = model.get_model_size()
        results["num_parameters"] = model.get_num_parameters()
    
    output_file = os.path.join(
        os.path.dirname(args.model_path),
        f"evaluation_results_{args.dataset}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory"
    )
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
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    main()
