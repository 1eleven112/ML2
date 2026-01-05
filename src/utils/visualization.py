"""
结果可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import os


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "training_curves.png"
):
    """
    绘制训练曲线
    
    Args:
        history: 包含训练历史的字典 (loss, accuracy等)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_ablation_results(
    results: Dict[str, Dict[str, float]],
    save_path: str = "ablation_results.png"
):
    """
    绘制消融实验结果
    
    Args:
        results: 消融实验结果字典
        save_path: 保存路径
    """
    # 准备数据
    models = list(results.keys())
    metrics = ['accuracy', 'compression_ratio', 'speedup']
    
    data = []
    for model_name, metrics_dict in results.items():
        data.append({
            'Model': model_name,
            'Accuracy (%)': metrics_dict.get('accuracy', 0) * 100,
            'Compression Ratio': metrics_dict.get('compression_ratio', 1.0),
            'Speedup': metrics_dict.get('speedup', 1.0),
        })
    
    df = pd.DataFrame(data)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 准确率对比
    axes[0].barh(df['Model'], df['Accuracy (%)'], color='skyblue')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 压缩比对比
    axes[1].barh(df['Model'], df['Compression Ratio'], color='lightcoral')
    axes[1].set_xlabel('Compression Ratio')
    axes[1].set_title('Model Compression Ratio')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 加速比对比
    axes[2].barh(df['Model'], df['Speedup'], color='lightgreen')
    axes[2].set_xlabel('Speedup')
    axes[2].set_title('Inference Speedup')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation results saved to {save_path}")


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    save_path: str = "comparison_table.csv"
):
    """
    创建模型对比表格
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
    """
    data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"Comparison table saved to {save_path}")
    
    return df


def plot_compression_tradeoff(
    results: Dict[str, Dict[str, float]],
    save_path: str = "compression_tradeoff.png"
):
    """
    绘制压缩率与性能的权衡图
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    accuracies = []
    compression_ratios = []
    
    for model_name, metrics in results.items():
        models.append(model_name)
        accuracies.append(metrics.get('accuracy', 0) * 100)
        compression_ratios.append(metrics.get('compression_ratio', 1.0))
    
    # 散点图
    scatter = ax.scatter(
        compression_ratios,
        accuracies,
        s=200,
        alpha=0.6,
        c=range(len(models)),
        cmap='viridis'
    )
    
    # 添加标签
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (compression_ratios[i], accuracies[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Compression-Performance Tradeoff', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Compression tradeoff plot saved to {save_path}")


def plot_attention_head_importance(
    head_importance: np.ndarray,
    save_path: str = "head_importance.png"
):
    """
    可视化注意力头重要性
    
    Args:
        head_importance: 注意力头重要性矩阵 [num_layers, num_heads]
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        head_importance,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Importance Score'},
        ax=ax
    )
    
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Layer')
    ax.set_title('Attention Head Importance Scores')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention head importance plot saved to {save_path}")


def generate_report(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "./results"
):
    """
    生成完整的实验报告
    
    Args:
        results: 实验结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各种图表
    plot_ablation_results(
        results,
        save_path=os.path.join(output_dir, "ablation_results.png")
    )
    
    plot_compression_tradeoff(
        results,
        save_path=os.path.join(output_dir, "compression_tradeoff.png")
    )
    
    create_comparison_table(
        results,
        save_path=os.path.join(output_dir, "comparison_table.csv")
    )
    
    print(f"Report generated in {output_dir}")
