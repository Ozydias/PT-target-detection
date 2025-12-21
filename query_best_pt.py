#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询 results.csv 中 best.pt 对应的数据
best.pt 通常对应 mAP50-95(B) 最高的那个 epoch
"""

import pandas as pd
import sys
from pathlib import Path

def query_best_pt_results(csv_path=None):
    """
    查询 best.pt 对应的训练结果
    
    Args:
        csv_path: results.csv 文件路径，如果为 None 则自动查找
    """
    # 如果没有指定路径，尝试自动查找
    if csv_path is None:
        # 尝试从当前目录查找
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "runs" / "detect" / "yolov12s_pt_v3" / "results.csv",
            current_dir / "results.csv",
            Path(__file__).parent / "runs" / "detect" / "yolov12s_pt_v3" / "results.csv",
        ]
        
        for path in possible_paths:
            if path.exists():
                csv_path = path
                break
        
        if csv_path is None:
            print("❌ 错误: 未找到 results.csv 文件")
            print("请指定文件路径，例如: python query_best_pt.py <path_to_results.csv>")
            return None
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ 错误: 文件不存在: {csv_path}")
        return None
    
    print("=" * 80)
    print("查询 best.pt 对应的训练结果")
    print("=" * 80)
    print(f"\n📁 文件路径: {csv_path}")
    
    # 读取 CSV 文件
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ 成功读取 CSV 文件")
        print(f"  - 原始数据行数: {len(df)}")
        
        # 检查是否有重复的 epoch
        if df['epoch'].duplicated().any():
            print(f"  ⚠️  检测到重复的 epoch 数据")
            # 对于每个 epoch，保留 mAP50-95 最高的那条记录
            map_column = 'metrics/mAP50-95(B)'
            if map_column in df.columns:
                # 按 epoch 分组，保留每组中 mAP50-95 最高的记录
                df = df.loc[df.groupby('epoch')[map_column].idxmax()].reset_index(drop=True)
                print(f"  ✓ 已处理重复数据，保留每个 epoch 中 mAP50-95 最高的记录")
            else:
                # 如果没有 mAP50-95 列，则保留最后一条
                df = df.drop_duplicates(subset=['epoch'], keep='last')
                print(f"  ✓ 已去除重复数据（保留最后一条）")
        
        df = df.sort_values('epoch').reset_index(drop=True)
        print(f"  - 处理后 epoch 数: {len(df)}")
        print(f"  - Epoch 范围: {int(df['epoch'].min())} - {int(df['epoch'].max())}")
    except Exception as e:
        print(f"❌ 读取 CSV 文件失败: {e}")
        return None
    
    # 找到 mAP50-95(B) 最高的 epoch（best.pt）
    map_column = 'metrics/mAP50-95(B)'
    if map_column not in df.columns:
        print(f"❌ 错误: 未找到列 '{map_column}'")
        print(f"可用列: {list(df.columns)}")
        return None
    
    # 找到最高 mAP50-95 的索引
    best_idx = df[map_column].idxmax()
    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_map = df.loc[best_idx, map_column]
    
    print(f"\n🏆 Best.pt 信息:")
    print(f"  - Epoch: {int(best_epoch)}")
    print(f"  - mAP50-95(B): {best_map:.6f}")
    
    # 显示 best.pt 的所有指标
    print(f"\n📊 Best.pt (Epoch {int(best_epoch)}) 的完整指标:")
    print("-" * 80)
    
    best_row = df.loc[best_idx]
    
    # 训练损失
    print("\n【训练损失】")
    if 'train/box_loss' in df.columns:
        print(f"  - Box Loss: {best_row['train/box_loss']:.6f}")
    if 'train/cls_loss' in df.columns:
        print(f"  - Class Loss: {best_row['train/cls_loss']:.6f}")
    if 'train/dfl_loss' in df.columns:
        print(f"  - DFL Loss: {best_row['train/dfl_loss']:.6f}")
    
    # 验证指标
    print("\n【验证指标】")
    if 'metrics/precision(B)' in df.columns:
        print(f"  - Precision: {best_row['metrics/precision(B)']:.6f}")
    if 'metrics/recall(B)' in df.columns:
        print(f"  - Recall: {best_row['metrics/recall(B)']:.6f}")
    if 'metrics/mAP50(B)' in df.columns:
        print(f"  - mAP50: {best_row['metrics/mAP50(B)']:.6f}")
    if 'metrics/mAP50-95(B)' in df.columns:
        print(f"  - mAP50-95: {best_row['metrics/mAP50-95(B)']:.6f}")
    
    # 验证损失
    print("\n【验证损失】")
    if 'val/box_loss' in df.columns:
        print(f"  - Box Loss: {best_row['val/box_loss']:.6f}")
    if 'val/cls_loss' in df.columns:
        print(f"  - Class Loss: {best_row['val/cls_loss']:.6f}")
    if 'val/dfl_loss' in df.columns:
        print(f"  - DFL Loss: {best_row['val/dfl_loss']:.6f}")
    
    # 学习率
    print("\n【学习率】")
    lr_columns = [col for col in df.columns if col.startswith('lr/')]
    for col in lr_columns:
        print(f"  - {col}: {best_row[col]:.8f}")
    
    # 训练时间
    if 'time' in df.columns:
        print(f"\n【训练时间】")
        print(f"  - 累计时间: {best_row['time']:.2f} 秒 ({best_row['time']/60:.2f} 分钟)")
    
    # 显示统计信息
    print(f"\n📈 统计信息:")
    print("-" * 80)
    print(f"  - 最高 mAP50-95: {df[map_column].max():.6f} (Epoch {int(df.loc[df[map_column].idxmax(), 'epoch'])})")
    print(f"  - 最低 mAP50-95: {df[map_column].min():.6f} (Epoch {int(df.loc[df[map_column].idxmin(), 'epoch'])})")
    print(f"  - 平均 mAP50-95: {df[map_column].mean():.6f}")
    print(f"  - 最终 mAP50-95: {df[map_column].iloc[-1]:.6f} (Epoch {int(df['epoch'].iloc[-1])})")
    
    # 显示前 5 个最好的 epoch
    print(f"\n🏅 Top 5 最佳 Epoch (按 mAP50-95 排序):")
    print("-" * 80)
    top5_cols = ['epoch', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 
                 'metrics/precision(B)', 'metrics/recall(B)']
    # 确保所有列都存在
    available_cols = [col for col in top5_cols if col in df.columns]
    top5 = df.nlargest(5, map_column)[available_cols]
    
    for idx, row in top5.iterrows():
        marker = " ⭐ BEST" if int(row['epoch']) == int(best_epoch) else ""
        epoch_str = f"Epoch {int(row['epoch']):3d}"
        map_str = f"mAP50-95={row['metrics/mAP50-95(B)']:.6f}"
        if 'metrics/mAP50(B)' in available_cols:
            map50_str = f"mAP50={row['metrics/mAP50(B)']:.6f}"
        else:
            map50_str = ""
        if 'metrics/precision(B)' in available_cols:
            prec_str = f"Precision={row['metrics/precision(B)']:.6f}"
        else:
            prec_str = ""
        if 'metrics/recall(B)' in available_cols:
            rec_str = f"Recall={row['metrics/recall(B)']:.6f}"
        else:
            rec_str = ""
        
        info_parts = [s for s in [map_str, map50_str, prec_str, rec_str] if s]
        print(f"  {epoch_str}: {', '.join(info_parts)}{marker}")
    
    print("\n" + "=" * 80)
    
    return best_row


if __name__ == '__main__':
    # 如果提供了命令行参数，使用它作为 CSV 文件路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = None
    
    result = query_best_pt_results(csv_path)
    
    if result is not None:
        print("✓ 查询完成！")
    else:
        sys.exit(1)

