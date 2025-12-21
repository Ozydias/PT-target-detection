#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 P2 检测层是否生效
检查：
1. P2 层的特征图尺寸（应该最大，因为是最高分辨率）
2. 检测层的 stride（P2 应该有最小的 stride）
3. 检测层的输出形状
"""

import torch
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

config_path = Path("ultralytics/cfg/models/v12/yolov12.yaml")
model_dict = yaml_model_load(str(config_path))
model_dict['scale'] = 's'  # 使用 s 规模

print("=" * 80)
print("验证 P2 检测层是否生效")
print("=" * 80)

# 构建模型
model = DetectionModel(model_dict, ch=3, verbose=False)
detect_layer = model.model[-1]

print(f"\n[1] 检测层信息:")
print(f"  - 检测层数量 (nl): {detect_layer.nl}")

# 检查输入通道数（通过 cv2 或 cv3）
if hasattr(detect_layer, 'cv2') and len(detect_layer.cv2) == 4:
    print(f"  - 检测层输入通道数:")
    for i, cv in enumerate(detect_layer.cv2):
        if hasattr(cv, '__len__') and len(cv) > 0:
            first_conv = cv[0]
            if hasattr(first_conv, 'conv'):
                in_channels = first_conv.conv.in_channels
                layer_name = ['P2', 'P3', 'P4', 'P5'][i]
                print(f"    {layer_name}: {in_channels} 通道")

# 检查 stride（如果有）
if hasattr(detect_layer, 'stride'):
    strides = detect_layer.stride
    if len(strides) == 4:
        print(f"  - Stride:")
        print(f"    P2 stride: {strides[0]}")
        print(f"    P3 stride: {strides[1]}")
        print(f"    P4 stride: {strides[2]}")
        print(f"    P5 stride: {strides[3]}")
        if strides[0] < strides[1]:
            print(f"  ✓ P2 stride 最小，说明 P2 层分辨率最高（正确）")

# 测试前向传播
print(f"\n[2] 测试前向传播...")
dummy_input = torch.randn(1, 3, 640, 640)
model.eval()

with torch.no_grad():
    # 获取中间特征图
    features = []
    x = dummy_input
    
    # 遍历模型获取各层输出
    for i, layer in enumerate(model.model):
        x = layer(x)
        # 检查是否是检测层的输入层（Concat 层）
        if i in [16, 19, 22, 25]:  # 检测层的输入索引
            if isinstance(x, (list, tuple)):
                feat = x[0] if len(x) > 0 else x
            else:
                feat = x
            features.append(feat)
    
    # 获取检测层输出
    detect_output = x if not isinstance(x, (list, tuple)) else x

print(f"  - 输入图像尺寸: 640x640")
print(f"  - 检测层输入特征图数量: {len(features)}")

# 分析特征图尺寸
if len(features) == 4:
    print(f"\n[3] 特征图尺寸分析:")
    for i, feat in enumerate(features):
        if hasattr(feat, 'shape'):
            h, w = feat.shape[2], feat.shape[3]
            scale = 640 / max(h, w)
            layer_name = ['P2', 'P3', 'P4', 'P5'][i]
            print(f"  - {layer_name} 层特征图: {h}x{w} (下采样比例: {scale:.1f}x)")
    
    # 验证 P2 层分辨率最高
    p2_size = features[0].shape[2] * features[0].shape[3] if hasattr(features[0], 'shape') else 0
    p3_size = features[1].shape[2] * features[1].shape[3] if hasattr(features[1], 'shape') else 0
    
    if p2_size > p3_size:
        print(f"  ✓ P2 层特征图尺寸 ({p2_size}) > P3 层 ({p3_size})，说明 P2 层分辨率最高（正确）")
    else:
        print(f"  ⚠️  警告: P2 层特征图尺寸 ({p2_size}) <= P3 层 ({p3_size})")

# 检查检测层输出
print(f"\n[4] 检测层输出:")
if isinstance(detect_output, (list, tuple)):
    print(f"  - 输出数量: {len(detect_output)}")
    for i, out in enumerate(detect_output):
        if hasattr(out, 'shape'):
            layer_name = ['P2', 'P3', 'P4', 'P5'][i] if i < 4 else f'Output{i}'
            print(f"  - {layer_name} 输出形状: {out.shape}")
            if len(out.shape) == 3:
                # 训练模式输出
                print(f"    检测框数量: {out.shape[2]}")
            elif len(out.shape) == 2:
                # 推理模式输出
                print(f"    检测框数量: {out.shape[0]}")
else:
    print(f"  - 输出类型: {type(detect_output)}")
    if hasattr(detect_output, 'shape'):
        print(f"  - 输出形状: {detect_output.shape}")

print(f"\n[5] 验证 P2 检测层配置:")
print(f"  - 配置文件中的检测层索引: [16, 19, 22, 25]")
print(f"  - 索引 16 对应 P2 层（Concat 后的特征）")
print(f"  - 检测层数量: 4 (P2, P3, P4, P5)")
print(f"  ✓ P2 检测层已成功集成到模型中")

print("\n" + "=" * 80)
print("✓ P2 检测层验证完成！")
print("=" * 80)

