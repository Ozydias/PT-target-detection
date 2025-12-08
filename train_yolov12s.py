#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 PTDataset 数据集训练 YOLOv12s 模型
"""

from ultralytics import YOLO
import os
from pathlib import Path
import yaml

def main():
    # 获取项目根目录（脚本所在目录）
    script_dir = Path(__file__).parent.absolute()
    
    # 检查数据集路径（使用绝对路径）
    dataset_path = script_dir / "PTDataset"
    if not dataset_path.exists():
        print(f"错误: 数据集目录 {dataset_path} 不存在！")
        return
    
    # 更新YAML配置文件中的路径为绝对路径
    yaml_path = script_dir / "PTDataset.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # 更新路径为绝对路径
        yaml_data['path'] = str(dataset_path)
        
        # 保存更新后的YAML文件
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ 已更新数据集路径为绝对路径: {dataset_path}")
    else:
        print(f"警告: 未找到 {yaml_path}，将创建新的配置文件")
        # 创建新的YAML文件
        yaml_data = {
            'path': str(dataset_path),
            'train': 'train',
            'val': 'val',
            'nc': 1,
            'names': {0: 'PT'}
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ 已创建数据集配置文件: {yaml_path}")
    
    # 检查标签文件是否存在
    train_labels = dataset_path / "labels" / "train"
    val_labels = dataset_path / "labels" / "val"
    
    # 清理标签目录中的图像文件（不应该存在）
    print("\n清理标签目录...")
    for label_dir in [train_labels, val_labels]:
        if label_dir.exists():
            jpg_files = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
            if jpg_files:
                print(f"  删除 {label_dir} 中的 {len(jpg_files)} 个图像文件...")
                for jpg_file in jpg_files:
                    jpg_file.unlink()
                print(f"  ✓ 已清理")
    
    if not train_labels.exists() or not val_labels.exists():
        print("警告: 标签文件目录不存在，可能需要先运行 convert_coco_to_yolo.py")
        print("正在检查是否需要转换...")
        
        # 检查是否有COCO格式的标注文件
        train_json = dataset_path / "annonations" / "instances_train.json"
        val_json = dataset_path / "annonations" / "instances_val.json"
        
        if train_json.exists() and val_json.exists():
            print("发现COCO格式标注文件，正在转换为YOLO格式...")
            from convert_coco_to_yolo import convert_coco_to_yolo
            convert_coco_to_yolo(dataset_dir=str(dataset_path))
        else:
            print("错误: 未找到标签文件或COCO格式标注文件！")
            return
    
    # 验证标签文件
    train_txt_files = list(train_labels.glob("*.txt")) if train_labels.exists() else []
    val_txt_files = list(val_labels.glob("*.txt")) if val_labels.exists() else []
    print(f"\n标签文件统计:")
    print(f"  训练集: {len(train_txt_files)} 个标签文件")
    print(f"  验证集: {len(val_txt_files)} 个标签文件")
    
    if len(train_txt_files) == 0:
        print("错误: 训练集没有标签文件！")
        return
    
    # 检查标签文件内容
    sample_label = train_txt_files[0]
    with open(sample_label, 'r') as f:
        content = f.read().strip()
    if not content:
        print(f"警告: 示例标签文件 {sample_label.name} 是空的")
    else:
        print(f"  示例标签文件: {sample_label.name}")
        print(f"  内容: {content.split(chr(10))[0] if chr(10) in content else content}")
    
    # 删除旧的缓存文件，强制重新生成
    cache_files = [
        dataset_path / "train.cache",
        dataset_path / "val.cache",
        dataset_path / "labels" / "train.cache",
        dataset_path / "labels" / "val.cache",
    ]
    print(f"\n清理缓存文件...")
    for cache_file in cache_files:
        if cache_file.exists():
            cache_file.unlink()
            print(f"  ✓ 已删除: {cache_file.name}")
    
    print("\n" + "=" * 60)
    print("开始训练 YOLOv12s 模型")
    print("=" * 60)
    print("\n重要提示:")
    print("  - 如果看到 'no labels found' 警告，说明数据集配置有问题")
    print("  - 训练损失应该从非零值开始，如果损失为0，说明没有加载标签")
    print("  - 第一个epoch的box_loss应该 > 0，如果为0，检查数据集配置")
    print("  - 验证指标会在训练过程中逐步提升")
    print("")
    
    # 检查之前的训练是否有问题
    old_results = script_dir / 'runs' / 'detect' / 'yolov12s_pt' / 'results.csv'
    if old_results.exists():
        print("⚠️  检测到之前的训练结果，如果损失为0，建议使用新的实验名称重新训练")
        print("")
    
    # 初始化模型 - 使用预训练权重（推荐，可以加速训练）
    # YOLO会自动下载预训练权重（如果本地不存在）
    print("\n加载 YOLOv12s 模型...")
    print("提示: 如果本地没有预训练权重，将自动从GitHub下载")
    model = YOLO('yolov12s.pt')
    print("✓ 成功加载模型")
    
    # 训练参数
    print("\n训练参数:")
    print("  模型: YOLOv12s")
    print("  数据集: PTDataset")
    print("  类别数: 1 (PT)")
    
    # 开始训练
    # 根据README，YOLOv12s推荐的训练参数：
    # scale=0.9, mixup=0.05, copy_paste=0.15
    results = model.train(
        data=str(yaml_path),        # 数据集配置文件（使用绝对路径）
        epochs=200,                  # 训练轮数（可根据需要调整，建议100-300）
        batch=16,                    # 批次大小（根据GPU内存调整：8GB GPU用8-16，16GB用16-32）
        imgsz=640,                   # 输入图像尺寸
        device='0',                  # 使用GPU 0（如果有多个GPU，可以用 '0,1,2,3'）
        project='runs/detect',       # 项目目录
        name='yolov12s_pt_v2',      # 新的实验名称（避免使用有问题的检查点）
        exist_ok=True,               # 允许覆盖已存在的实验
        pretrained=True,             # 使用预训练权重
        optimizer='AdamW',           # 优化器
        verbose=True,                # 显示详细信息
        seed=42,                     # 随机种子
        deterministic=True,          # 确定性训练
        # 学习率设置（针对单类别数据集优化）
        lr0=0.01,                    # 初始学习率
        lrf=0.1,                     # 最终学习率 (lr0 * lrf)
        momentum=0.937,              # SGD动量（如果使用SGD）
        weight_decay=0.0005,         # 权重衰减
        warmup_epochs=3.0,           # 预热轮数
        # YOLOv12s 推荐参数
        scale=0.9,                   # S模型推荐scale=0.9
        mixup=0.05,                  # S模型推荐mixup=0.05
        copy_paste=0.15,             # S模型推荐copy_paste=0.15
        mosaic=1.0,                  # Mosaic数据增强
        # 其他常用参数
        cos_lr=False,                # 余弦学习率调度
        close_mosaic=10,             # 最后10个epoch关闭mosaic
        resume=False,                # 是否从检查点恢复
        amp=True,                    # 自动混合精度训练
        val=True,                    # 训练期间验证
        # 验证参数调整（降低阈值以确保能检测到目标）
        conf=0.0001,                 # 置信度阈值（进一步降低，确保能检测到目标）
        iou=0.5,                     # IoU阈值（降低以提高召回率）
        max_det=300,                 # 每张图像最大检测数量
        plots=True,                  # 保存验证结果图表
        # 损失权重调整（针对单类别优化）
        box=7.5,                     # 边界框损失权重
        cls=0.5,                     # 分类损失权重（单类别可以降低）
        dfl=1.5,                     # DFL损失权重
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 评估模型
    print("\n评估模型性能...")
    metrics = model.val()
    print(f"\n验证集 mAP50: {metrics.box.map50:.4f}")
    print(f"验证集 mAP50-95: {metrics.box.map:.4f}")
    
    # 保存最佳模型路径
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    print(f"\n最佳模型保存在: {best_model_path}")
    
    return model, results


if __name__ == '__main__':
    model, results = main()

