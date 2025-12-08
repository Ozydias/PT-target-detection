#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据集配置和标签文件
"""

from pathlib import Path
import yaml

def check_dataset():
    script_dir = Path(__file__).parent.absolute()
    dataset_path = script_dir / "PTDataset"
    yaml_path = script_dir / "PTDataset.yaml"
    
    print("=" * 60)
    print("数据集诊断工具")
    print("=" * 60)
    
    # 1. 检查数据集目录
    print(f"\n1. 检查数据集目录: {dataset_path}")
    if not dataset_path.exists():
        print(f"   ❌ 数据集目录不存在: {dataset_path}")
        return
    print(f"   ✓ 数据集目录存在")
    
    # 2. 检查YAML配置文件
    print(f"\n2. 检查YAML配置文件: {yaml_path}")
    if not yaml_path.exists():
        print(f"   ❌ YAML配置文件不存在")
        return
    print(f"   ✓ YAML配置文件存在")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    print(f"   配置内容:")
    for key, value in yaml_data.items():
        print(f"     {key}: {value}")
    
    # 3. 检查图像文件
    train_images = dataset_path / "train"
    val_images = dataset_path / "val"
    
    print(f"\n3. 检查图像文件")
    if train_images.exists():
        train_img_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        print(f"   ✓ 训练图像目录存在，找到 {len(train_img_files)} 个图像文件")
        if len(train_img_files) > 0:
            print(f"     示例: {train_img_files[0].name}")
    else:
        print(f"   ❌ 训练图像目录不存在: {train_images}")
    
    if val_images.exists():
        val_img_files = list(val_images.glob("*.jpg")) + list(val_images.glob("*.png"))
        print(f"   ✓ 验证图像目录存在，找到 {len(val_img_files)} 个图像文件")
        if len(val_img_files) > 0:
            print(f"     示例: {val_img_files[0].name}")
    else:
        print(f"   ❌ 验证图像目录不存在: {val_images}")
    
    # 4. 检查标签文件
    train_labels = dataset_path / "labels" / "train"
    val_labels = dataset_path / "labels" / "val"
    
    print(f"\n4. 检查标签文件")
    if train_labels.exists():
        train_txt_files = list(train_labels.glob("*.txt"))
        train_jpg_files = list(train_labels.glob("*.jpg"))  # 不应该有jpg文件
        print(f"   ✓ 训练标签目录存在")
        print(f"     找到 {len(train_txt_files)} 个 .txt 标签文件")
        if len(train_jpg_files) > 0:
            print(f"     ⚠️  警告: 标签目录中有 {len(train_jpg_files)} 个 .jpg 文件（不应该存在）")
        
        # 检查标签文件内容
        if len(train_txt_files) > 0:
            sample_label = train_txt_files[0]
            with open(sample_label, 'r') as f:
                content = f.read().strip()
                lines = content.split('\n') if content else []
            print(f"     示例标签文件: {sample_label.name}")
            print(f"     内容行数: {len(lines)}")
            if len(lines) > 0:
                print(f"     第一行: {lines[0]}")
                # 验证格式
                parts = lines[0].split()
                if len(parts) == 5:
                    try:
                        class_id, cx, cy, w, h = map(float, parts)
                        print(f"     ✓ 格式正确: class={int(class_id)}, center=({cx:.4f}, {cy:.4f}), size=({w:.4f}, {h:.4f})")
                        if class_id < 0 or class_id >= yaml_data.get('nc', 1):
                            print(f"     ⚠️  警告: 类别ID {int(class_id)} 超出范围 [0, {yaml_data.get('nc', 1)-1}]")
                    except ValueError:
                        print(f"     ❌ 格式错误: 无法解析数字")
                else:
                    print(f"     ❌ 格式错误: 应该有5个值，实际有{len(parts)}个")
    else:
        print(f"   ❌ 训练标签目录不存在: {train_labels}")
    
    if val_labels.exists():
        val_txt_files = list(val_labels.glob("*.txt"))
        print(f"   ✓ 验证标签目录存在")
        print(f"     找到 {len(val_txt_files)} 个 .txt 标签文件")
    else:
        print(f"   ❌ 验证标签目录不存在: {val_labels}")
    
    # 5. 检查图像和标签文件匹配
    print(f"\n5. 检查图像和标签文件匹配")
    if train_images.exists() and train_labels.exists():
        train_img_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        train_txt_files = list(train_labels.glob("*.txt"))
        
        img_names = {f.stem for f in train_img_files}
        txt_names = {f.stem for f in train_txt_files}
        
        missing_labels = img_names - txt_names
        missing_images = txt_names - img_names
        
        print(f"   训练集:")
        print(f"     图像文件数: {len(img_names)}")
        print(f"     标签文件数: {len(txt_names)}")
        if missing_labels:
            print(f"     ⚠️  警告: {len(missing_labels)} 个图像没有对应的标签文件")
            print(f"       示例: {list(missing_labels)[:5]}")
        if missing_images:
            print(f"     ⚠️  警告: {len(missing_images)} 个标签文件没有对应的图像文件")
            print(f"       示例: {list(missing_images)[:5]}")
        if not missing_labels and not missing_images:
            print(f"     ✓ 所有图像都有对应的标签文件")
    
    # 6. 检查空标签文件
    print(f"\n6. 检查空标签文件")
    if train_labels.exists():
        train_txt_files = list(train_labels.glob("*.txt"))
        empty_files = []
        for txt_file in train_txt_files:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    empty_files.append(txt_file.name)
        
        if empty_files:
            print(f"     ⚠️  警告: 找到 {len(empty_files)} 个空标签文件")
            print(f"       示例: {empty_files[:5]}")
        else:
            print(f"     ✓ 所有标签文件都有内容")
    
    # 7. 检查标签文件中的类别ID
    print(f"\n7. 检查标签文件中的类别ID")
    if train_labels.exists():
        train_txt_files = list(train_labels.glob("*.txt"))
        class_ids = set()
        invalid_class_ids = []
        
        for txt_file in train_txt_files[:100]:  # 只检查前100个文件
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(float(parts[0]))
                                class_ids.add(class_id)
                                if class_id < 0 or class_id >= yaml_data.get('nc', 1):
                                    invalid_class_ids.append((txt_file.name, class_id))
                            except ValueError:
                                pass
        
        print(f"   检查了 {min(100, len(train_txt_files))} 个标签文件")
        print(f"   发现的类别ID: {sorted(class_ids)}")
        if invalid_class_ids:
            print(f"     ❌ 错误: 发现无效的类别ID")
            for filename, cid in invalid_class_ids[:5]:
                print(f"       {filename}: 类别ID {cid} (应该在 [0, {yaml_data.get('nc', 1)-1}] 范围内)")
        else:
            print(f"     ✓ 所有类别ID都在有效范围内 [0, {yaml_data.get('nc', 1)-1}]")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    check_dataset()

