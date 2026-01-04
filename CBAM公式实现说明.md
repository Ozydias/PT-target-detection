# CBAM注意力机制公式 Fout = Fin ⊗ Ac ⊗ As 代码实现说明

## 公式含义
- **Fin**: 输入特征图 (Input Feature)
- **Ac**: 通道注意力权重 (Channel Attention Weights)
- **As**: 空间注意力权重 (Spatial Attention Weights)
- **Fout**: 输出特征图 (Output Feature)
- **⊗**: 逐元素相乘 (Element-wise Multiplication)

## 代码实现位置

**文件：`ultralytics/nn/modules/attention.py`**

**类：CBAM (第42-56行)**

### 完整代码实现：

```python
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = SEModule(channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 第一步：Fout' = Fin ⊗ Ac (通道注意力)
        x_out = self.channel_attention(x)  # ← 这里实现 Fin ⊗ Ac
        
        # 第二步：计算空间注意力的输入（对通道注意力输出进行池化）
        avg_out = torch.mean(x_out, dim=1, keepdim=True)
        max_out = torch.max(x_out, dim=1, keepdim=True)[0]
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))  # 计算 As
        
        # 第三步：Fout = Fout' ⊗ As = (Fin ⊗ Ac) ⊗ As
        return x_out * sa  # ← 这里实现 Fout = Fin ⊗ Ac ⊗ As
```

## 公式对应关系详解

### 第52行：`x_out = self.channel_attention(x)`
**对应公式：** `Fout' = Fin ⊗ Ac`

- `x` 是输入特征 `Fin`
- `self.channel_attention(x)` 计算通道注意力权重 `Ac` 并与 `Fin` 逐元素相乘
- `x_out` 是经过通道注意力处理后的特征 `Fout'`

### 第56行：`return x_out * sa`
**对应公式：** `Fout = Fout' ⊗ As = (Fin ⊗ Ac) ⊗ As`

- `x_out` 是 `Fin ⊗ Ac` 的结果
- `sa` 是空间注意力权重 `As`（由第55行计算得到）
- `x_out * sa` 实现了最终公式：`Fout = Fin ⊗ Ac ⊗ As`

## 完整计算流程

```
输入特征 Fin
    ↓
[通道注意力模块]
    ↓
Fin ⊗ Ac (第52行：x_out = self.channel_attention(x))
    ↓
[空间注意力模块的输入准备]
    ↓
计算空间注意力权重 As (第55行：sa = self.spatial_attention(...))
    ↓
Fin ⊗ Ac ⊗ As (第56行：return x_out * sa)
    ↓
输出特征 Fout
```

## PPT截图建议

**截图位置：`ultralytics/nn/modules/attention.py` 第42-56行**

特别标注：
- 第52行：用红色框标注 `x_out = self.channel_attention(x)` ← 实现 `Fin ⊗ Ac`
- 第56行：用蓝色框标注 `return x_out * sa` ← 实现 `Fout = Fin ⊗ Ac ⊗ As`
- 可以在代码上方添加注释说明公式对应关系



