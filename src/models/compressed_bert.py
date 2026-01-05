"""
压缩BERT模型定义
支持混合压缩方法：知识蒸馏、剪枝、量化
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Optional, Dict, List


class CompressedBertConfig:
    """压缩BERT配置"""
    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_size: int = 768,
        # 剪枝配置
        pruning_ratio: float = 0.4,
        head_pruning_ratio: float = 0.3,
        ffn_pruning_ratio: float = 0.25,
        # 量化配置
        quantization_bits: int = 8,
        quantize_embeddings: bool = False,
        # 蒸馏配置
        use_distillation: bool = True,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.5,
    ):
        self.base_model_name = base_model_name
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        
        self.pruning_ratio = pruning_ratio
        self.head_pruning_ratio = head_pruning_ratio
        self.ffn_pruning_ratio = ffn_pruning_ratio
        
        self.quantization_bits = quantization_bits
        self.quantize_embeddings = quantize_embeddings
        
        self.use_distillation = use_distillation
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class CompressedBertForSequenceClassification(nn.Module):
    """
    支持混合压缩的BERT序列分类模型
    """
    def __init__(
        self,
        config: CompressedBertConfig,
        num_labels: int = 2,
        from_pretrained: bool = True
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # 加载基础BERT模型
        if from_pretrained:
            self.bert = BertModel.from_pretrained(config.base_model_name)
        else:
            bert_config = BertConfig.from_pretrained(config.base_model_name)
            self.bert = BertModel(bert_config)
        
        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # 用于存储注意力头和FFN的重要性分数
        self.head_importance = None
        self.ffn_importance = None
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            token_type_ids: token类型IDs [batch_size, seq_length]
            labels: 标签 [batch_size]
            return_hidden_states: 是否返回隐藏状态
            return_attentions: 是否返回注意力权重
            
        Returns:
            包含loss、logits、hidden_states、attentions的字典
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_hidden_states,
            output_attentions=return_attentions,
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        result = {
            "loss": loss,
            "logits": logits,
        }
        
        if return_hidden_states:
            result["hidden_states"] = outputs.hidden_states
        
        if return_attentions:
            result["attentions"] = outputs.attentions
            
        return result
    
    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """获取模型参数量"""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size(self) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def save_compressed_model(self, output_dir: str):
        """保存压缩后的模型"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # 保存配置
        config_dict = {
            "num_labels": self.num_labels,
            "config": {
                "base_model_name": self.config.base_model_name,
                "num_hidden_layers": self.config.num_hidden_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "hidden_size": self.config.hidden_size,
                "intermediate_size": self.config.intermediate_size,
            }
        }
        
        import json
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Model saved to {output_dir}")
        print(f"Model size: {self.get_model_size():.2f} MB")
        print(f"Number of parameters: {self.get_num_parameters() / 1e6:.2f}M")
    
    @classmethod
    def load_compressed_model(cls, model_dir: str):
        """加载压缩后的模型"""
        import os
        import json
        
        # 加载配置
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # 创建配置对象
        config = CompressedBertConfig(
            base_model_name=config_dict["config"]["base_model_name"],
            num_hidden_layers=config_dict["config"]["num_hidden_layers"],
            num_attention_heads=config_dict["config"]["num_attention_heads"],
            hidden_size=config_dict["config"]["hidden_size"],
            intermediate_size=config_dict["config"]["intermediate_size"],
        )
        
        # 创建模型
        model = cls(config, num_labels=config_dict["num_labels"], from_pretrained=False)
        
        # 加载权重
        state_dict = torch.load(
            os.path.join(model_dir, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        
        return model
