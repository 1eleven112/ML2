"""
分析和可视化消融实验结果
"""

import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import (
    plot_ablation_results,
    plot_compression_tradeoff,
    create_comparison_table
)


def collect_results(results_dir: str, dataset: str = "sst2"):
    """
    收集所有实验结果
    
    Args:
        results_dir: 结果目录
        dataset: 数据集名称
        
    Returns:
        结果字典
    """
    results = {}
    
    experiment_names = [
        'baseline',
        'kd_only',
        'pruning_only',
        'quantization_only',
        'kd_pruning',
        'kd_quantization',
        'pruning_quantization',
        'hybrid_full'
    ]
    
    display_names = {
        'baseline': 'Baseline',
        'kd_only': 'KD only',
        'pruning_only': 'Pruning only',
        'quantization_only': 'Quantization only',
        'kd_pruning': 'KD + Pruning',
        'kd_quantization': 'KD + Quantization',
        'pruning_quantization': 'Pruning + Quantization',
        'hybrid_full': 'Hybrid (Full)'
    }
    
    baseline_size = None
    baseline_time = None
    
    for exp_name in experiment_names:
        result_file = os.path.join(
            results_dir,
            exp_name,
            f"evaluation_results_{dataset}.json"
        )
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # 提取关键指标
            model_size = data.get('model_size_mb', 438.0)
            inference_time = data.get('inference_time_ms', 45.0)
            accuracy = data['val_metrics'].get('accuracy', 0)
            
            if exp_name == 'baseline':
                baseline_size = model_size
                baseline_time = inference_time
            
            compression_ratio = baseline_size / model_size if baseline_size and model_size > 0 else 1.0
            speedup = baseline_time / inference_time if baseline_time and inference_time > 0 else 1.0
            
            results[display_names[exp_name]] = {
                'accuracy': accuracy,
                'model_size_mb': model_size,
                'inference_time_ms': inference_time,
                'compression_ratio': compression_ratio,
                'speedup': speedup,
                'params': data.get('num_parameters', 0),
            }
            
            print(f"Loaded results for {display_names[exp_name]}")
    
    return results


def generate_latex_table(results: dict, output_file: str):
    """
    生成LaTeX格式的表格
    
    Args:
        results: 结果字典
        output_file: 输出文件路径
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{消融实验结果对比}")
    lines.append("\\label{tab:ablation_results}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append("模型 & 准确率(\\%) & 参数量 & 模型大小 & 推理时间(ms) & 压缩比 & 加速比 \\\\")
    lines.append("\\hline")
    
    for model_name, metrics in results.items():
        acc = metrics['accuracy'] * 100
        params = f"{metrics.get('params', 0) / 1e6:.1f}M"
        size = f"{metrics['model_size_mb']:.0f}MB"
        time = f"{metrics['inference_time_ms']:.1f}"
        comp = f"{metrics['compression_ratio']:.2f}x"
        speed = f"{metrics['speedup']:.2f}x"
        
        lines.append(f"{model_name} & {acc:.2f} & {params} & {size} & {time} & {comp} & {speed} \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to {output_file}")


def generate_report(results_dir: str, dataset: str = "sst2", output_dir: str = None):
    """
    生成完整的实验报告
    
    Args:
        results_dir: 结果目录
        dataset: 数据集名称
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "report")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Collecting results...")
    results = collect_results(results_dir, dataset)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nFound {len(results)} experiments")
    
    # 生成对比表格
    print("\nGenerating comparison table...")
    df = create_comparison_table(
        results,
        save_path=os.path.join(output_dir, "comparison_table.csv")
    )
    print(df.to_string(index=False))
    
    # 生成可视化图表
    print("\nGenerating visualizations...")
    plot_ablation_results(
        results,
        save_path=os.path.join(output_dir, "ablation_results.png")
    )
    
    plot_compression_tradeoff(
        results,
        save_path=os.path.join(output_dir, "compression_tradeoff.png")
    )
    
    # 生成LaTeX表格
    print("\nGenerating LaTeX table...")
    generate_latex_table(
        results,
        output_file=os.path.join(output_dir, "results_table.tex")
    )
    
    # 生成Markdown报告
    print("\nGenerating Markdown report...")
    generate_markdown_report(results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Report generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # 打印关键发现
    print("\n关键发现:")
    
    if 'Baseline' in results and 'Hybrid (Full)' in results:
        baseline_acc = results['Baseline']['accuracy']
        hybrid_acc = results['Hybrid (Full)']['accuracy']
        acc_drop = (baseline_acc - hybrid_acc) * 100
        compression = results['Hybrid (Full)']['compression_ratio']
        speedup = results['Hybrid (Full)']['speedup']
        
        print(f"1. 混合压缩方法:")
        print(f"   - 压缩比: {compression:.2f}x")
        print(f"   - 加速比: {speedup:.2f}x")
        print(f"   - 准确率下降: {acc_drop:.2f}%")
        print(f"   - 最终准确率: {hybrid_acc*100:.2f}%")


def generate_markdown_report(results: dict, output_dir: str):
    """生成Markdown格式的报告"""
    lines = []
    lines.append("# 消融实验结果报告\n")
    lines.append("## 实验结果汇总\n")
    
    # 创建表格
    lines.append("| 模型 | 准确率(%) | 参数量 | 模型大小 | 推理时间(ms) | 压缩比 | 加速比 |")
    lines.append("|------|-----------|--------|----------|--------------|--------|--------|")
    
    for model_name, metrics in results.items():
        acc = metrics['accuracy'] * 100
        params = f"{metrics.get('params', 0) / 1e6:.1f}M"
        size = f"{metrics['model_size_mb']:.0f}MB"
        time = f"{metrics['inference_time_ms']:.1f}"
        comp = f"{metrics['compression_ratio']:.2f}x"
        speed = f"{metrics['speedup']:.2f}x"
        
        lines.append(f"| {model_name} | {acc:.2f} | {params} | {size} | {time} | {comp} | {speed} |")
    
    lines.append("\n## 分析结论\n")
    
    # 计算一些统计信息
    if 'Baseline' in results:
        baseline_acc = results['Baseline']['accuracy']
        
        lines.append("### 单一压缩方法效果\n")
        
        if 'KD only' in results:
            kd_acc = results['KD only']['accuracy']
            lines.append(f"- **知识蒸馏**: 准确率 {kd_acc*100:.2f}% (下降 {(baseline_acc-kd_acc)*100:.2f}%)")
        
        if 'Pruning only' in results:
            prune_acc = results['Pruning only']['accuracy']
            prune_comp = results['Pruning only']['compression_ratio']
            lines.append(f"- **剪枝**: 准确率 {prune_acc*100:.2f}% (下降 {(baseline_acc-prune_acc)*100:.2f}%), 压缩比 {prune_comp:.2f}x")
        
        if 'Quantization only' in results:
            quant_acc = results['Quantization only']['accuracy']
            quant_comp = results['Quantization only']['compression_ratio']
            lines.append(f"- **量化**: 准确率 {quant_acc*100:.2f}% (下降 {(baseline_acc-quant_acc)*100:.2f}%), 压缩比 {quant_comp:.2f}x")
        
        lines.append("\n### 组合压缩方法效果\n")
        
        if 'Hybrid (Full)' in results:
            hybrid_acc = results['Hybrid (Full)']['accuracy']
            hybrid_comp = results['Hybrid (Full)']['compression_ratio']
            hybrid_speed = results['Hybrid (Full)']['speedup']
            lines.append(f"- **混合方法**: 准确率 {hybrid_acc*100:.2f}% (下降 {(baseline_acc-hybrid_acc)*100:.2f}%)")
            lines.append(f"  - 压缩比: {hybrid_comp:.2f}x")
            lines.append(f"  - 加速比: {hybrid_speed:.2f}x")
            lines.append(f"  - 效率提升: {(hybrid_comp * hybrid_speed) / 2:.2f}x")
    
    with open(os.path.join(output_dir, "REPORT.md"), 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Markdown report saved to {os.path.join(output_dir, 'REPORT.md')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation experiment results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/ablation",
        help="Results directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        help="Dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for report"
    )
    
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
