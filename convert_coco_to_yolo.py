#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将COCO格式的JSON标注文件转换为YOLO格式
适用于PTDataset数据集
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np


def convert_coco_to_yolo(
    dataset_dir="PTDataset",
    output_dir=None,
    train_json="annonations/instances_train.json",
    val_json="annonations/instances_val.json"
):
    """
    将COCO格式的标注转换为YOLO格式
    
    Args:
        dataset_dir: 数据集根目录
        output_dir: 输出目录（如果为None，则在原目录创建labels文件夹）
        train_json: 训练集JSON文件路径（相对于dataset_dir）
        val_json: 验证集JSON文件路径（相对于dataset_dir）
    """
    dataset_path = Path(dataset_dir)
    
    if output_dir is None:
        output_dir = dataset_path
    else:
        output_dir = Path(output_dir)
    
    # 创建输出目录结构
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取类别映射（从JSON文件中获取）
    train_json_path = dataset_path / train_json
    val_json_path = dataset_path / val_json
    
    print("=" * 60)
    print("COCO格式转YOLO格式转换工具")
    print("=" * 60)
    
    # 处理训练集
    print(f"\n处理训练集: {train_json_path}")
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 创建类别ID到YOLO类别ID的映射（COCO类别ID从1开始，YOLO从0开始）
    categories = {cat['id']: cat for cat in train_data['categories']}
    cat_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
    
    print(f"类别映射:")
    for coco_id, yolo_id in cat_id_to_yolo.items():
        print(f"  COCO ID {coco_id} ({categories[coco_id]['name']}) -> YOLO ID {yolo_id}")
    
    # 创建图像ID到图像信息的映射
    images = {img['id']: img for img in train_data['images']}
    
    # 创建图像ID到标注列表的映射
    img_to_anns = defaultdict(list)
    for ann in train_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # 转换训练集标注
    train_count = 0
    for img_id, anns in img_to_anns.items():
        img = images[img_id]
        h, w = img['height'], img['width']
        img_name = img['file_name']
        
        # YOLO格式标签文件路径
        label_file = labels_train_dir / (Path(img_name).stem + '.txt')
        
        # 转换标注
        yolo_labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            # COCO格式: [x_min, y_min, width, height] (绝对坐标)
            bbox = np.array(ann['bbox'], dtype=np.float32)
            
            # 转换为YOLO格式: [class_id, center_x, center_y, width, height] (归一化坐标)
            x_min, y_min, bbox_w, bbox_h = bbox
            
            # 计算中心点坐标
            center_x = (x_min + bbox_w / 2) / w
            center_y = (y_min + bbox_h / 2) / h
            norm_w = bbox_w / w
            norm_h = bbox_h / h
            
            # 获取YOLO类别ID
            yolo_cls = cat_id_to_yolo[ann['category_id']]
            
            # 检查边界
            if norm_w <= 0 or norm_h <= 0:
                continue
            if center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
                continue
            
            yolo_labels.append(f"{yolo_cls} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # 写入标签文件
        if yolo_labels:
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_labels))
            train_count += 1
    
    print(f"训练集: 转换了 {train_count} 个标签文件")
    
    # 处理验证集
    print(f"\n处理验证集: {val_json_path}")
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    images = {img['id']: img for img in val_data['images']}
    img_to_anns = defaultdict(list)
    for ann in val_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    val_count = 0
    for img_id, anns in img_to_anns.items():
        img = images[img_id]
        h, w = img['height'], img['width']
        img_name = img['file_name']
        
        label_file = labels_val_dir / (Path(img_name).stem + '.txt')
        
        yolo_labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            bbox = np.array(ann['bbox'], dtype=np.float32)
            x_min, y_min, bbox_w, bbox_h = bbox
            
            center_x = (x_min + bbox_w / 2) / w
            center_y = (y_min + bbox_h / 2) / h
            norm_w = bbox_w / w
            norm_h = bbox_h / h
            
            yolo_cls = cat_id_to_yolo[ann['category_id']]
            
            if norm_w <= 0 or norm_h <= 0:
                continue
            if center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
                continue
            
            yolo_labels.append(f"{yolo_cls} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        if yolo_labels:
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_labels))
            val_count += 1
    
    print(f"验证集: 转换了 {val_count} 个标签文件")
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"标签文件保存在:")
    print(f"  训练集: {labels_train_dir}")
    print(f"  验证集: {labels_val_dir}")
    print(f"\n现在可以使用 PTDataset.yaml 配置文件进行训练了！")
    
    return labels_train_dir, labels_val_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将COCO格式标注转换为YOLO格式')
    parser.add_argument('--dataset_dir', type=str, default='PTDataset',
                        help='数据集根目录 (默认: PTDataset)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: 在数据集目录下创建labels文件夹)')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )

