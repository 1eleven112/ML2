"""
模型剪枝模块
实现结构化剪枝：注意力头剪枝和FFN维度剪枝
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from transformers import BertModel


class BertPruner:
    """
    BERT模型剪枝器
    支持注意力头剪枝和FFN维度剪枝
    """
    def __init__(
        self,
        model: nn.Module,
        head_pruning_ratio: float = 0.3,
        ffn_pruning_ratio: float = 0.25,
    ):
        """
        Args:
            model: 待剪枝的BERT模型
            head_pruning_ratio: 注意力头剪枝比例
            ffn_pruning_ratio: FFN维度剪枝比例
        """
        self.model = model
        self.head_pruning_ratio = head_pruning_ratio
        self.ffn_pruning_ratio = ffn_pruning_ratio
        
        # 获取BERT编码器
        if hasattr(model, 'bert'):
            self.bert = model.bert
        else:
            self.bert = model
            
        self.config = self.bert.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        
        # 存储重要性分数
        self.head_importance = None
        self.ffn_importance = None
        
    def compute_head_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        num_samples: int = 1000,
    ) -> torch.Tensor:
        """
        计算注意力头的重要性分数
        使用梯度的L1范数作为重要性指标
        
        Args:
            dataloader: 数据加载器
            device: 设备
            num_samples: 用于评估的样本数量
            
        Returns:
            注意力头重要性矩阵 [num_layers, num_heads]
        """
        self.model.eval()
        head_importance = torch.zeros(self.num_layers, self.num_heads).to(device)
        head_mask = torch.ones(self.num_layers, self.num_heads).to(device)
        head_mask.requires_grad_(True)
        
        num_processed = 0
        for batch in dataloader:
            if num_processed >= num_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs['loss']
            loss.backward()
            
            # 累积梯度的L1范数
            head_importance += head_mask.grad.abs().detach()
            head_mask.grad = None
            
            num_processed += input_ids.size(0)
        
        # 归一化
        head_importance = head_importance / num_processed
        self.head_importance = head_importance
        
        return head_importance
    
    def compute_ffn_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        num_samples: int = 1000,
    ) -> List[torch.Tensor]:
        """
        计算FFN中间层维度的重要性分数
        
        Args:
            dataloader: 数据加载器
            device: 设备
            num_samples: 用于评估的样本数量
            
        Returns:
            每层FFN维度重要性列表
        """
        self.model.eval()
        ffn_importance = []
        
        # 为每层的FFN收集激活值
        for layer_idx in range(self.num_layers):
            intermediate_size = self.config.intermediate_size
            importance = torch.zeros(intermediate_size).to(device)
            
            def hook_fn(module, input, output):
                # 累积激活值的L1范数
                importance.add_(output.abs().mean(dim=(0, 1)))
            
            # 注册hook
            layer = self.bert.encoder.layer[layer_idx]
            handle = layer.intermediate.dense.register_forward_hook(hook_fn)
            
            num_processed = 0
            with torch.no_grad():
                for batch in dataloader:
                    if num_processed >= num_samples:
                        break
                        
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    
                    num_processed += input_ids.size(0)
            
            handle.remove()
            importance = importance / num_processed
            ffn_importance.append(importance)
        
        self.ffn_importance = ffn_importance
        return ffn_importance
    
    def prune_heads(
        self,
        head_importance: torch.Tensor,
    ) -> Dict[int, List[int]]:
        """
        根据重要性分数剪枝注意力头
        
        Args:
            head_importance: 注意力头重要性矩阵 [num_layers, num_heads]
            
        Returns:
            每层要剪枝的注意力头索引字典 {layer_idx: [head_indices]}
        """
        heads_to_prune = {}
        
        for layer_idx in range(self.num_layers):
            layer_importance = head_importance[layer_idx]
            num_heads_to_prune = int(self.num_heads * self.head_pruning_ratio)
            
            if num_heads_to_prune > 0:
                # 选择重要性最低的头进行剪枝
                _, indices = torch.topk(
                    layer_importance,
                    k=num_heads_to_prune,
                    largest=False
                )
                heads_to_prune[layer_idx] = indices.cpu().tolist()
        
        # 执行剪枝
        self.bert.encoder.prune_heads(heads_to_prune)
        
        return heads_to_prune
    
    def prune_ffn_dimensions(
        self,
        ffn_importance: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        根据重要性分数剪枝FFN维度
        
        Args:
            ffn_importance: 每层FFN维度重要性列表
            
        Returns:
            每层保留的维度索引列表
        """
        kept_indices = []
        
        for layer_idx, importance in enumerate(ffn_importance):
            num_dims_to_keep = int(
                self.config.intermediate_size * (1 - self.ffn_pruning_ratio)
            )
            
            # 选择重要性最高的维度保留
            _, indices = torch.topk(
                importance,
                k=num_dims_to_keep,
                largest=True
            )
            kept_indices.append(indices.sort()[0])  # 保持顺序
        
        # 执行FFN剪枝（需要修改权重矩阵）
        for layer_idx, indices in enumerate(kept_indices):
            layer = self.bert.encoder.layer[layer_idx]
            
            # 剪枝intermediate层
            intermediate_weight = layer.intermediate.dense.weight.data[indices]
            intermediate_bias = layer.intermediate.dense.bias.data[indices]
            
            new_intermediate_size = len(indices)
            layer.intermediate.dense = nn.Linear(
                self.config.hidden_size,
                new_intermediate_size
            )
            layer.intermediate.dense.weight.data = intermediate_weight
            layer.intermediate.dense.bias.data = intermediate_bias
            
            # 剪枝output层
            output_weight = layer.output.dense.weight.data[:, indices]
            layer.output.dense = nn.Linear(
                new_intermediate_size,
                self.config.hidden_size
            )
            layer.output.dense.weight.data = output_weight
            layer.output.dense.bias.data = layer.output.dense.bias.data
        
        return kept_indices
    
    def apply_pruning(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> Dict[str, any]:
        """
        应用完整的剪枝流程
        
        Args:
            dataloader: 数据加载器
            device: 设备
            
        Returns:
            剪枝统计信息
        """
        print("Computing attention head importance...")
        head_importance = self.compute_head_importance(dataloader, device)
        
        print("Computing FFN dimension importance...")
        ffn_importance = self.compute_ffn_importance(dataloader, device)
        
        print("Pruning attention heads...")
        heads_pruned = self.prune_heads(head_importance)
        
        print("Pruning FFN dimensions...")
        ffn_kept_indices = self.prune_ffn_dimensions(ffn_importance)
        
        # 统计信息
        total_heads = self.num_layers * self.num_heads
        pruned_heads = sum(len(v) for v in heads_pruned.values())
        
        total_ffn_dims = self.num_layers * self.config.intermediate_size
        kept_ffn_dims = sum(len(indices) for indices in ffn_kept_indices)
        
        stats = {
            "heads_pruned": pruned_heads,
            "total_heads": total_heads,
            "head_pruning_ratio": pruned_heads / total_heads,
            "ffn_dims_kept": kept_ffn_dims,
            "total_ffn_dims": total_ffn_dims,
            "ffn_pruning_ratio": 1 - (kept_ffn_dims / total_ffn_dims),
            "heads_pruned_per_layer": heads_pruned,
        }
        
        print(f"\nPruning Statistics:")
        print(f"  Heads pruned: {pruned_heads}/{total_heads} "
              f"({stats['head_pruning_ratio']:.2%})")
        print(f"  FFN dims kept: {kept_ffn_dims}/{total_ffn_dims} "
              f"({1 - stats['ffn_pruning_ratio']:.2%})")
        
        return stats
