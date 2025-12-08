#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复数据集问题：删除标签目录中的图像文件
"""

from pathlib import Path

def fix_dataset():
    script_dir = Path(__file__).parent.absolute()
    dataset_path = script_dir / "PTDataset"
    
    print("=" * 60)
    print("修复数据集")
    print("=" * 60)
    
    # 删除标签目录中的图像文件
    train_labels = dataset_path / "labels" / "train"
    val_labels = dataset_path / "labels" / "val"
    
    deleted_count = 0
    
    if train_labels.exists():
        jpg_files = list(train_labels.glob("*.jpg")) + list(train_labels.glob("*.png"))
        print(f"\n训练标签目录中发现 {len(jpg_files)} 个图像文件")
        for jpg_file in jpg_files:
            jpg_file.unlink()
            deleted_count += 1
        print(f"✓ 已删除 {len(jpg_files)} 个图像文件")
    
    if val_labels.exists():
        jpg_files = list(val_labels.glob("*.jpg")) + list(val_labels.glob("*.png"))
        print(f"\n验证标签目录中发现 {len(jpg_files)} 个图像文件")
        for jpg_file in jpg_files:
            jpg_file.unlink()
            deleted_count += 1
        print(f"✓ 已删除 {len(jpg_files)} 个图像文件")
    
    print(f"\n总共删除了 {deleted_count} 个图像文件")
    print("=" * 60)
    print("修复完成！现在可以重新运行训练脚本。")
    print("=" * 60)

if __name__ == '__main__':
    fix_dataset()



