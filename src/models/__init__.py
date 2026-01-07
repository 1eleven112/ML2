"""
模型模块
包含压缩BERT模型和各种压缩技术的实现
"""

from .compressed_bert import CompressedBertForSequenceClassification, CompressedBertConfig
from .distillation import DistillationLoss, DistillationTrainer, create_distillation_trainer
from .pruning import BertPruner
from .quantization import (
    QuantizationConfig,
    ModelQuantizer,
    QuantizedLinear,
    quantize_model
)

__all__ = [
    'CompressedBertForSequenceClassification',
    'CompressedBertConfig',
    'DistillationLoss',
    'DistillationTrainer',
    'create_distillation_trainer',
    'BertPruner',
    'QuantizationConfig',
    'ModelQuantizer',
    'QuantizedLinear',
    'quantize_model',
]
