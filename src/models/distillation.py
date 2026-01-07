"""
知识蒸馏模块
实现教师-学生模型的知识转移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    包含：输出层蒸馏、中间层特征蒸馏、注意力蒸馏
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        """
        Args:
            temperature: 蒸馏温度，用于软化概率分布
            alpha: CE损失权重
            beta: KD损失权重
            gamma: 特征蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden_states: Optional[torch.Tensor] = None,
        teacher_hidden_states: Optional[torch.Tensor] = None,
        student_attentions: Optional[torch.Tensor] = None,
        teacher_attentions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch_size, num_labels]
            teacher_logits: 教师模型输出 [batch_size, num_labels]
            labels: 真实标签 [batch_size]
            student_hidden_states: 学生模型隐藏状态 (可选)
            teacher_hidden_states: 教师模型隐藏状态 (可选)
            student_attentions: 学生模型注意力权重 (可选)
            teacher_attentions: 教师模型注意力权重 (可选)
            
        Returns:
            包含总损失和各项子损失的字典
        """
        # 1. 交叉熵损失（硬标签）
        ce_loss = self.ce_loss(student_logits, labels)
        
        # 2. KL散度损失（软标签）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 3. 中间层特征蒸馏损失
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_hidden_states is not None and teacher_hidden_states is not None:
            # 选择最后几层进行特征蒸馏
            num_layers = min(len(student_hidden_states), len(teacher_hidden_states))
            for i in range(max(0, num_layers - 4), num_layers):
                student_hidden = student_hidden_states[i]
                teacher_hidden = teacher_hidden_states[i]
                
                # 如果维度不同，使用线性变换对齐
                if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
                    # 简单的均值池化或插值
                    teacher_hidden = F.adaptive_avg_pool1d(
                        teacher_hidden.transpose(1, 2),
                        student_hidden.shape[-1]
                    ).transpose(1, 2)
                
                feature_loss += self.mse_loss(student_hidden, teacher_hidden)
            
            feature_loss /= 4  # 平均损失
        
        # 4. 注意力蒸馏损失
        attention_loss = torch.tensor(0.0, device=student_logits.device)
        if student_attentions is not None and teacher_attentions is not None:
            num_layers = min(len(student_attentions), len(teacher_attentions))
            for i in range(num_layers):
                # 注意力矩阵的MSE损失
                attention_loss += self.mse_loss(
                    student_attentions[i],
                    teacher_attentions[i]
                )
            attention_loss /= num_layers
        
        # 总损失
        total_loss = (
            self.alpha * ce_loss +
            self.beta * kd_loss +
            self.gamma * (feature_loss + attention_loss)
        )
        
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
            "feature_loss": feature_loss,
            "attention_loss": attention_loss,
        }


class DistillationTrainer:
    """
    知识蒸馏训练器
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        """
        Args:
            teacher_model: 教师模型（已训练好的大模型）
            student_model: 学生模型（待训练的小模型）
            temperature: 蒸馏温度
            alpha: CE损失权重
            beta: KD损失权重
            gamma: 特征蒸馏损失权重
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_loss = DistillationLoss(temperature, alpha, beta, gamma)
        
        # 冻结教师模型
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签
            token_type_ids: token类型IDs (可选)
            
        Returns:
            损失字典
        """
        # 教师模型推理（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_hidden_states=True,
                return_attentions=True,
            )
        
        # 学生模型推理
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_hidden_states=True,
            return_attentions=True,
        )
        
        # 计算蒸馏损失
        loss_dict = self.distillation_loss(
            student_logits=student_outputs["logits"],
            teacher_logits=teacher_outputs["logits"],
            labels=labels,
            student_hidden_states=student_outputs.get("hidden_states"),
            teacher_hidden_states=teacher_outputs.get("hidden_states"),
            student_attentions=student_outputs.get("attentions"),
            teacher_attentions=teacher_outputs.get("attentions"),
        )
        
        return loss_dict


def create_distillation_trainer(
    teacher_model_name: str,
    student_model: nn.Module,
    device: str = "cuda",
    temperature: float = 2.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> DistillationTrainer:
    """
    创建知识蒸馏训练器
    
    Args:
        teacher_model_name: 教师模型名称或路径
        student_model: 学生模型
        device: 设备
        temperature: 蒸馏温度
        alpha: CE损失权重
        beta: KD损失权重
        gamma: 特征蒸馏损失权重
        
    Returns:
        DistillationTrainer实例
    """
    from transformers import AutoModelForSequenceClassification
    
    # 加载教师模型
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_name
    )
    teacher_model.to(device)
    
    # 移动学生模型到设备
    student_model.to(device)
    
    # 创建训练器
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    
    return trainer
