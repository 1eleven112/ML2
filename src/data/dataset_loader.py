"""
GLUE数据集加载器
支持多种GLUE任务的数据加载和预处理
"""

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Tuple, Optional


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
    
    # 确定要删除的列（保留label列）
    columns_to_remove = [col for col in dataset['train'].column_names if col != 'label']
    
    # 对数据集进行编码
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    
    # 重命名label列为labels（符合transformers约定）
    if 'label' in encoded_dataset['train'].column_names:
        encoded_dataset = encoded_dataset.rename_column('label', 'labels')
    
    # 设置格式
    encoded_dataset.set_format(type='torch')
    
    # 验证集 - 某些GLUE任务使用validation，其他使用validation_matched
    if 'validation' in encoded_dataset:
        val_dataset = encoded_dataset['validation']
    elif 'validation_matched' in encoded_dataset:
        val_dataset = encoded_dataset['validation_matched']
    else:
        # 如果没有验证集，从训练集中分割（避免数据泄露）
        from datasets import DatasetDict
        train_val_split = encoded_dataset['train'].train_test_split(test_size=0.1, seed=42)
        encoded_dataset = DatasetDict({
            'train': train_val_split['train'],
            'validation': train_val_split['test'],
        })
        val_dataset = encoded_dataset['validation']
    
    # 创建数据加载器
    train_loader = DataLoader(
        encoded_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    print(f"  Val samples: {len(val_dataset)}")
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
