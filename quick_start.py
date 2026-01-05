"""
快速开始示例
演示如何使用本项目进行模型压缩
"""

import torch
from transformers import AutoTokenizer
from src.models.compressed_bert import CompressedBertForSequenceClassification, CompressedBertConfig
from src.data.dataset_loader import load_glue_data
from src.utils.metrics import evaluate_model


def quick_start_example():
    """快速开始示例"""
    
    print("="*60)
    print("Transformer Model Compression Quick Start")
    print("="*60)
    
    # 1. 创建压缩BERT配置
    print("\n1. Creating compressed BERT configuration...")
    config = CompressedBertConfig(
        base_model_name="bert-base-uncased",
        pruning_ratio=0.4,
        head_pruning_ratio=0.3,
        ffn_pruning_ratio=0.25,
        quantization_bits=8,
        use_distillation=True,
    )
    
    # 2. 创建模型
    print("\n2. Creating compressed model...")
    model = CompressedBertForSequenceClassification(
        config=config,
        num_labels=2,
        from_pretrained=True
    )
    
    print(f"   Model size: {model.get_model_size():.2f} MB")
    print(f"   Parameters: {model.get_num_parameters() / 1e6:.2f}M")
    
    # 3. 加载数据
    print("\n3. Loading SST-2 dataset...")
    train_loader, val_loader, _, _ = load_glue_data(
        dataset_name="sst2",
        batch_size=8,  # Small batch for demo
        max_length=128,
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # 4. 简单推理示例
    print("\n4. Running inference on a sample...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    text = "This movie is absolutely amazing!"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        prediction = outputs['logits'].argmax(dim=-1).item()
        probs = torch.softmax(outputs['logits'], dim=-1)
    
    label = "Positive" if prediction == 1 else "Negative"
    confidence = probs[0, prediction].item()
    
    print(f"   Text: {text}")
    print(f"   Prediction: {label}")
    print(f"   Confidence: {confidence:.4f}")
    
    # 5. 训练示例（演示，不实际训练）
    print("\n5. Training setup example...")
    print("   To train the model, run:")
    print("   python src/train_compressed.py \\")
    print("       --teacher_model bert-base-uncased \\")
    print("       --dataset sst2 \\")
    print("       --compression_methods distillation pruning quantization \\")
    print("       --output_dir ./results/my_model")
    
    print("\n" + "="*60)
    print("Quick start completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train a baseline model: python src/train_baseline.py")
    print("2. Train a compressed model: python src/train_compressed.py")
    print("3. Run ablation experiments: bash experiments/run_ablation.sh")
    print("4. Evaluate a model: python src/evaluate.py --model_path <path>")
    print("="*60)


if __name__ == "__main__":
    quick_start_example()
