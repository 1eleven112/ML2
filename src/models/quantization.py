"""
模型量化模块
实现INT8量化和混合精度量化
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class QuantizationConfig:
    """量化配置"""
    def __init__(
        self,
        bits: int = 8,
        quantize_embeddings: bool = False,
        quantize_classifier: bool = True,
        symmetric: bool = True,
        per_channel: bool = True,
    ):
        self.bits = bits
        self.quantize_embeddings = quantize_embeddings
        self.quantize_classifier = quantize_classifier
        self.symmetric = symmetric
        self.per_channel = per_channel


class LinearQuantizer:
    """
    线性层量化器
    实现对称/非对称量化
    """
    def __init__(self, bits: int = 8, symmetric: bool = True, per_channel: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.qmin = -2 ** (bits - 1) if symmetric else 0
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
        
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        dim: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        量化张量
        
        Args:
            tensor: 待量化的张量
            dim: 按通道量化时的维度
            
        Returns:
            (量化后的张量, 缩放因子, 零点)
        """
        if dim is not None and self.per_channel:
            # 按通道量化
            shape = [1] * tensor.dim()
            shape[dim] = -1
            
            min_val = tensor.amin(dim=tuple(i for i in range(tensor.dim()) if i != dim), keepdim=True)
            max_val = tensor.amax(dim=tuple(i for i in range(tensor.dim()) if i != dim), keepdim=True)
        else:
            # 按张量量化
            min_val = tensor.min()
            max_val = tensor.max()
        
        if self.symmetric:
            # 对称量化
            abs_max = torch.max(min_val.abs(), max_val.abs())
            scale = abs_max / (2 ** (self.bits - 1) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            # 非对称量化
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            zero_point = -min_val / scale
        
        # 避免除零
        scale = torch.clamp(scale, min=1e-8)
        
        # 量化
        q_tensor = torch.clamp(
            torch.round(tensor / scale + zero_point),
            self.qmin,
            self.qmax
        )
        
        return q_tensor, scale, zero_point
    
    def dequantize_tensor(
        self,
        q_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        反量化张量
        
        Args:
            q_tensor: 量化后的张量
            scale: 缩放因子
            zero_point: 零点
            
        Returns:
            反量化后的张量
        """
        return (q_tensor - zero_point) * scale


class QuantizedLinear(nn.Module):
    """
    量化的线性层
    在前向传播时执行量化和反量化
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        self.quantizer = LinearQuantizer(bits, symmetric, per_channel)
        
        # 权重和偏置
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 量化参数（将在校准时设置）
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_zero_point', torch.zeros(1))
        self.register_buffer('input_scale', torch.ones(1))
        self.register_buffer('input_zero_point', torch.zeros(1))
        
        self.calibrated = False
        
    def calibrate(self, weight: Optional[torch.Tensor] = None):
        """
        校准量化参数
        
        Args:
            weight: 可选的权重张量（从预训练模型复制）
        """
        if weight is not None:
            self.weight.data = weight
        
        # 量化权重
        q_weight, scale, zero_point = self.quantizer.quantize_tensor(
            self.weight.data, dim=0
        )
        
        self.weight_scale = scale
        self.weight_zero_point = zero_point
        self.calibrated = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（量化感知训练）
        
        Args:
            x: 输入张量 [batch_size, ..., in_features]
            
        Returns:
            输出张量 [batch_size, ..., out_features]
        """
        if self.training:
            # 训练时：模拟量化
            q_weight, _, _ = self.quantizer.quantize_tensor(self.weight, dim=0)
            dq_weight = self.quantizer.dequantize_tensor(
                q_weight, self.weight_scale, self.weight_zero_point
            )
        else:
            # 推理时：使用校准的参数
            if not self.calibrated:
                self.calibrate()
            dq_weight = self.weight
        
        return nn.functional.linear(x, dq_weight, self.bias)


class ModelQuantizer:
    """
    模型量化器
    将模型中的线性层替换为量化版本
    """
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        量化模型
        
        Args:
            model: 待量化的模型
            
        Returns:
            量化后的模型
        """
        # 复制模型
        quantized_model = copy.deepcopy(model)
        
        # 递归替换线性层
        self._replace_linear_layers(quantized_model)
        
        return quantized_model
    
    def _replace_linear_layers(self, module: nn.Module, name: str = ""):
        """
        递归替换模型中的线性层为量化版本
        
        Args:
            module: 模块
            name: 模块名称
        """
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear):
                # 检查是否需要量化该层
                should_quantize = True
                
                if "embeddings" in full_name and not self.config.quantize_embeddings:
                    should_quantize = False
                    
                if "classifier" in full_name and not self.config.quantize_classifier:
                    should_quantize = False
                
                if should_quantize:
                    # 创建量化层
                    quantized_layer = QuantizedLinear(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bits=self.config.bits,
                        symmetric=self.config.symmetric,
                        per_channel=self.config.per_channel,
                        bias=child_module.bias is not None,
                    )
                    
                    # 复制权重
                    quantized_layer.weight.data = child_module.weight.data.clone()
                    if child_module.bias is not None:
                        quantized_layer.bias.data = child_module.bias.data.clone()
                    
                    # 校准
                    quantized_layer.calibrate()
                    
                    # 替换
                    setattr(module, child_name, quantized_layer)
            else:
                # 递归处理子模块
                self._replace_linear_layers(child_module, full_name)
    
    def calibrate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
        num_batches: int = 10,
    ):
        """
        校准量化模型
        使用一些样本数据统计激活值的范围
        
        Args:
            model: 量化后的模型
            dataloader: 校准数据加载器
            device: 设备
            num_batches: 校准批次数
        """
        model.eval()
        model.to(device)
        
        print("Calibrating quantized model...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        
        print("Calibration completed.")


def quantize_model(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cuda",
) -> nn.Module:
    """
    便捷函数：量化模型
    
    Args:
        model: 待量化的模型
        config: 量化配置
        dataloader: 校准数据加载器（可选）
        device: 设备
        
    Returns:
        量化后的模型
    """
    if config is None:
        config = QuantizationConfig()
    
    quantizer = ModelQuantizer(config)
    quantized_model = quantizer.quantize_model(model)
    
    if dataloader is not None:
        quantizer.calibrate_model(quantized_model, dataloader, device)
    
    return quantized_model
