"""
GLUE数据集加载器
支持多种GLUE任务的数据加载和预处理
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Tuple, Optional


class GLUEDataset(Dataset):
    """GLUE数据集包装器"""
    
    def __init__(self, encodings, labels):
        """
        Args:
            encodings: 编码后的输入
            labels: 标签
        """
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# GLUE任务配置
GLUE_TASK_CONFIG = {
    'sst2': {
        'dataset_name': 'glue',
        'task_name': 'sst2',
        'num_labels': 2,
        'text_fields': ['sentence'],
        'metric': 'accuracy',
    },
    'mrpc': {
        'dataset_name': 'glue',
        'task_name': 'mrpc',
        'num_labels': 2,
        'text_fields': ['sentence1', 'sentence2'],
        'metric': 'f1',
    },
    'cola': {
        'dataset_name': 'glue',
        'task_name': 'cola',
        'num_labels': 2,
        'text_fields': ['sentence'],
        'metric': 'matthews_correlation',
    },
    'qnli': {
        'dataset_name': 'glue',
        'task_name': 'qnli',
        'num_labels': 2,
        'text_fields': ['question', 'sentence'],
        'metric': 'accuracy',
    },
    'qqp': {
        'dataset_name': 'glue',
        'task_name': 'qqp',
        'num_labels': 2,
        'text_fields': ['question1', 'question2'],
        'metric': 'f1',
    },
    'rte': {
        'dataset_name': 'glue',
        'task_name': 'rte',
        'num_labels': 2,
        'text_fields': ['sentence1', 'sentence2'],
        'metric': 'accuracy',
    },
}


def preprocess_function(examples, tokenizer, text_fields, max_length):
    """
    预处理GLUE数据集
    
    Args:
        examples: 数据样本
        tokenizer: 分词器
        text_fields: 文本字段列表
        max_length: 最大序列长度
        
    Returns:
        编码后的数据
    """
    if len(text_fields) == 1:
        # 单句任务
        texts = examples[text_fields[0]]
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
    elif len(text_fields) == 2:
        # 句对任务
        texts1 = examples[text_fields[0]]
        texts2 = examples[text_fields[1]]
        return tokenizer(
            texts1,
            texts2,
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unsupported number of text fields: {len(text_fields)}")


def load_glue_data(
    dataset_name: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], int]:
    """
    加载GLUE数据集
    
    Args:
        dataset_name: GLUE任务名称 (sst2, mrpc, cola, qnli, qqp, rte)
        model_name: 预训练模型名称，用于加载对应的分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: DataLoader工作线程数
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器 (如果可用)
        num_labels: 标签数量
        
    Example:
        >>> train_loader, val_loader, test_loader, num_labels = load_glue_data(
        ...     dataset_name="sst2",
        ...     batch_size=32,
        ...     max_length=128
        ... )
    """
    # 检查任务是否支持
    if dataset_name.lower() not in GLUE_TASK_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {list(GLUE_TASK_CONFIG.keys())}"
        )
    
    task_config = GLUE_TASK_CONFIG[dataset_name.lower()]
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载数据集
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(
        task_config['dataset_name'],
        task_config['task_name']
    )
    
    # 预处理数据
    text_fields = task_config['text_fields']
    
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, text_fields, max_length)
    
    # 对数据集进行编码
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # 设置格式
    encoded_dataset.set_format(type='torch')
    
    # 创建数据加载器
    train_loader = DataLoader(
        encoded_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    # 验证集 - 某些GLUE任务使用validation，其他使用validation_matched
    if 'validation' in encoded_dataset:
        val_split = 'validation'
    elif 'validation_matched' in encoded_dataset:
        val_split = 'validation_matched'
    else:
        # 如果没有验证集，使用训练集的一部分
        val_split = 'train'
    
    val_loader = DataLoader(
        encoded_dataset[val_split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    # 测试集 - 某些GLUE任务有测试集
    test_loader = None
    if 'test' in encoded_dataset:
        test_loader = DataLoader(
            encoded_dataset['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    num_labels = task_config['num_labels']
    
    print(f"Dataset loaded:")
    print(f"  Train samples: {len(encoded_dataset['train'])}")
    print(f"  Val samples: {len(encoded_dataset[val_split])}")
    if test_loader is not None:
        print(f"  Test samples: {len(encoded_dataset['test'])}")
    print(f"  Number of labels: {num_labels}")
    
    return train_loader, val_loader, test_loader, num_labels


def get_dataset_info(dataset_name: str) -> dict:
    """
    获取数据集信息
    
    Args:
        dataset_name: GLUE任务名称
        
    Returns:
        数据集配置信息
    """
    if dataset_name.lower() not in GLUE_TASK_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {list(GLUE_TASK_CONFIG.keys())}"
        )
    
    return GLUE_TASK_CONFIG[dataset_name.lower()]
