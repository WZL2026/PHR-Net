import os
import random
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader


def create_kfold_dataloaders(train_dataset, val_dataset, batch_size=2, num_workers=1, k=5,
                             worker_init_fn=None):
    """
    创建五折验证的数据加载器

    Args:
        train_dataset: 训练数据集（带有数据增强）
        val_dataset: 验证数据集（没有数据增强，保持原图）
        batch_size: 批量大小
        num_workers: 数据加载线程数
        k: 折数
        worker_init_fn: DataLoader worker 随机种子初始化函数

    Returns:
        生成器，每次返回 (train_loader, val_loader, fold_idx)
    """
    # 创建 KFold 分割器
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # 获取数据集索引
    indices = list(range(len(train_dataset)))

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        # 创建训练和验证子集
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        yield train_loader, val_loader, fold_idx


def get_dataset_name(dataset):
    """
    从数据集对象获取数据集名称
    """
    dataset_class_name = dataset.__class__.__name__
    if 'DRIVE' in dataset_class_name:
        return 'DRIVE'
    elif 'CHASE' in dataset_class_name:
        return 'CHASEDB1'
    elif 'STARE' in dataset_class_name:
        return 'STARE'
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class_name}")
