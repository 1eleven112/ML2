"""
评估指标模块
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from typing import Dict, List


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """计算准确率"""
    return accuracy_score(labels, predictions)


def compute_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """计算F1分数"""
    return f1_score(labels, predictions, average='binary')


def compute_matthews_correlation(predictions: np.ndarray, labels: np.ndarray) -> float:
    """计算Matthews相关系数"""
    return matthews_corrcoef(labels, predictions)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "sst2"
) -> Dict[str, float]:
    """
    根据数据集计算相应的评估指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        dataset_name: 数据集名称
        
    Returns:
        指标字典
    """
    metrics = {}
    
    if dataset_name in ["sst2", "qnli", "rte", "wnli"]:
        metrics["accuracy"] = compute_accuracy(predictions, labels)
    elif dataset_name == "mrpc":
        metrics["accuracy"] = compute_accuracy(predictions, labels)
        metrics["f1"] = compute_f1(predictions, labels)
    elif dataset_name == "cola":
        metrics["matthews_correlation"] = compute_matthews_correlation(predictions, labels)
    elif dataset_name == "qqp":
        metrics["accuracy"] = compute_accuracy(predictions, labels)
        metrics["f1"] = compute_f1(predictions, labels)
    else:
        # 默认计算准确率
        metrics["accuracy"] = compute_accuracy(predictions, labels)
    
    return metrics


class MetricsTracker:
    """指标跟踪器"""
    def __init__(self, dataset_name: str = "sst2"):
        self.dataset_name = dataset_name
        self.reset()
    
    def reset(self):
        """重置跟踪器"""
        self.predictions = []
        self.labels = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor = None
    ):
        """
        更新跟踪器
        
        Args:
            predictions: 预测logits或类别
            labels: 真实标签
            loss: 损失值
        """
        # 如果是logits，取argmax
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())
        
        if loss is not None:
            self.losses.append(loss.item())
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = compute_metrics(predictions, labels, self.dataset_name)
        
        if self.losses:
            metrics["loss"] = np.mean(self.losses)
        
        return metrics
    
    def get_primary_metric(self) -> str:
        """获取主要指标名称"""
        if self.dataset_name == "cola":
            return "matthews_correlation"
        elif self.dataset_name == "mrpc":
            return "f1"
        else:
            return "accuracy"


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    dataset_name: str = "sst2",
) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 设备
        dataset_name: 数据集名称
        
    Returns:
        评估指标字典
    """
    model.eval()
    tracker = MetricsTracker(dataset_name)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            token_type_ids = None
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            
            tracker.update(
                predictions=outputs['logits'],
                labels=labels,
                loss=outputs.get('loss')
            )
    
    return tracker.compute()
