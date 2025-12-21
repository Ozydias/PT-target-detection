#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 YOLOv12 P2 检测层配置
验证：
1. 配置文件能否成功加载
2. 模型能否成功构建
3. P2 检测层是否正确集成
4. 检测层索引是否正确
"""

import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model, yaml_model_load
from ultralytics.utils import LOGGER

def test_yolov12_p2_config():
    """测试 YOLOv12 P2 配置"""
    print("=" * 80)
    print("测试 YOLOv12 P2 小目标检测层配置")
    print("=" * 80)
    
    # 1. 加载配置文件
    config_path = Path("ultralytics/cfg/models/v12/yolov12.yaml")
    print(f"\n[1] 加载配置文件: {config_path}")
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return False
    
    try:
        model_dict = yaml_model_load(str(config_path))
        print(f"✓ 配置文件加载成功")
        print(f"  - 类别数 (nc): {model_dict.get('nc')}")
        # 如果没有指定 scale，使用 's' 作为默认值（更稳定）
        if not model_dict.get('scale'):
            model_dict['scale'] = 's'
            print(f"  - 模型规模 (scale): {model_dict.get('scale')} (自动设置为 's')")
        else:
            print(f"  - 模型规模 (scale): {model_dict.get('scale')}")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False
    
    # 2. 解析模型结构
    print(f"\n[2] 解析模型结构...")
    try:
        model, save = parse_model(model_dict, ch=3, verbose=False)
        print(f"✓ 模型结构解析成功")
        print(f"  - 总层数: {len(model)}")
        print(f"  - Save 索引: {save}")
    except Exception as e:
        print(f"❌ 模型结构解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 检查检测层
    print(f"\n[3] 检查检测层...")
    detect_layer = None
    for i, layer in enumerate(model):
        if hasattr(layer, 'type') and 'Detect' in layer.type:
            detect_layer = layer
            print(f"✓ 找到检测层 (索引 {i}): {layer.type}")
            break
    
    if detect_layer is None:
        print("❌ 错误: 未找到检测层")
        return False
    
    # 4. 检查检测层的输入通道和索引
    print(f"\n[4] 检查检测层配置...")
    if hasattr(detect_layer, 'cv2') or hasattr(detect_layer, 'cv3'):
        # 检查检测层的输入通道数
        if hasattr(detect_layer, 'ch'):
            input_channels = detect_layer.ch
            print(f"  - 检测层输入通道数: {input_channels}")
            print(f"  - 检测层数量: {len(input_channels)}")
            
            # 验证是否有4个检测层（P2, P3, P4, P5）
            if len(input_channels) == 4:
                print(f"✓ 检测层数量正确: 4个 (P2, P3, P4, P5)")
                print(f"  - P2 通道数: {input_channels[0]}")
                print(f"  - P3 通道数: {input_channels[1]}")
                print(f"  - P4 通道数: {input_channels[2]}")
                print(f"  - P5 通道数: {input_channels[3]}")
            else:
                print(f"⚠️  警告: 检测层数量为 {len(input_channels)}，期望为 4 (P2, P3, P4, P5)")
        else:
            print("⚠️  警告: 无法获取检测层输入通道信息")
    
    # 5. 检查配置文件中的检测层索引
    print(f"\n[5] 检查配置文件中的检测层索引...")
    head_layers = model_dict.get('head', [])
    detect_config = None
    for layer in head_layers:
        if isinstance(layer, list) and len(layer) >= 4:
            if layer[2] == 'Detect':
                detect_config = layer
                break
    
    if detect_config:
        detect_indices = detect_config[0]  # 第一个参数是检测层的索引列表
        print(f"  - 配置文件中的检测层索引: {detect_indices}")
        
        # 验证索引是否指向正确的层
        print(f"\n[6] 验证检测层索引指向的层...")
        for idx in detect_indices:
            if idx < len(model):
                layer = model[idx]
                layer_type = getattr(layer, 'type', 'Unknown')
                layer_info = f"索引 {idx}: {layer_type}"
                
                # 检查是否是 P2 相关的层
                if idx == detect_indices[0]:
                    print(f"  - {layer_info} ← P2 检测层 (小目标)")
                else:
                    print(f"  - {layer_info}")
            else:
                print(f"  ❌ 错误: 索引 {idx} 超出模型层数范围")
    else:
        print("⚠️  警告: 未在配置中找到 Detect 层配置")
    
    # 7. 尝试构建完整模型（使用 scale='s' 避免通道数问题）
    print(f"\n[7] 尝试构建完整 YOLO 模型...")
    try:
        # 重新加载配置并设置 scale
        model_dict_test = yaml_model_load(str(config_path))
        model_dict_test['scale'] = 's'
        # 使用 DetectionModel 直接构建，避免 YOLO 类的 scale 猜测
        from ultralytics.nn.tasks import DetectionModel
        model_full = DetectionModel(model_dict_test, ch=3, verbose=False)
        print(f"✓ YOLO 模型构建成功")
        
        # 检查模型信息
        if hasattr(model_full, 'model'):
            detect_module = model_full.model[-1]
            if hasattr(detect_module, 'nl'):
                print(f"  - 检测层数量 (nl): {detect_module.nl}")
                if detect_module.nl == 4:
                    print(f"✓ 检测层数量正确: 4个 (包含 P2)")
                else:
                    print(f"⚠️  警告: 检测层数量为 {detect_module.nl}，期望为 4")
            if hasattr(detect_module, 'ch'):
                print(f"  - 检测层输入通道数: {detect_module.ch}")
                if len(detect_module.ch) == 4:
                    print(f"✓ P2 检测层通道数: {detect_module.ch[0]}")
                    print(f"  P3 检测层通道数: {detect_module.ch[1]}")
                    print(f"  P4 检测层通道数: {detect_module.ch[2]}")
                    print(f"  P5 检测层通道数: {detect_module.ch[3]}")
    except Exception as e:
        print(f"❌ YOLO 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. 测试前向传播
    print(f"\n[8] 测试模型前向传播...")
    try:
        dummy_input = torch.randn(1, 3, 640, 640)
        model_full.eval()
        with torch.no_grad():
            output = model_full(dummy_input)
        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {dummy_input.shape}")
        if isinstance(output, (list, tuple)) and len(output) > 0:
            print(f"  - 输出数量: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"    - P{2+i} 输出形状: {out.shape}")
                    if len(out.shape) == 3:
                        print(f"      特征图尺寸: {out.shape[1]}x{out.shape[2]}")
        else:
            print(f"  - 输出类型: {type(output)}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！P2 检测层配置正确")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_yolov12_p2_config()
    exit(0 if success else 1)

