import torch
import torch.nn as nn
import torch.nn.functional as F
class PurePyTorchMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        # 修改点 1：严格使用 Mamba 官方标准的 padding=3 (kernel_size - 1)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 记住原始的序列长度 (这就是报错里的那个 400)
        L = x.shape[1]

        x_proj = self.in_proj(x)
        x_conv, z = x_proj.chunk(2, dim=-1)

        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv1d(x_conv)

        # 修改点 2：【终极一刀】！切掉多出来的尾巴，强行对齐回 400！
        x_conv = x_conv[..., :L]

        x_conv = x_conv.transpose(1, 2)

        # 现在 400 和 400 严丝合缝，再也不会报错了！
        x_out = F.silu(x_conv) * F.silu(z)

        out = self.out_proj(x_out)
        return out


# ============== 军师 1:1 逆向重构：Mamba 空间握手模块 ==============
class MambaHandshake(nn.Module):
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        # 1. 挂载 Mamba 核心模块
        self.mamba = PurePyTorchMambaBlock(c1)

        # 2. 对应报告: model.11.heatmap_gen
        self.heatmap_gen = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 路线 A：提取 Mamba 全局特征 ---
        x_flat = x.flatten(2).transpose(1, 2)  # 展平为序列 (B, H*W, C)
        mamba_out = self.mamba(x_flat)
        mamba_out = mamba_out.transpose(1, 2).reshape(B, C, H, W)  # 变回图片形状

        # --- 路线 B：生成空间注意力热力图 (Heatmap) ---
        heatmap = self.heatmap_gen(x)
        attention = torch.sigmoid(heatmap)

        # --- 终极握手 (Handshake)：残差 + 局部注意力加权 ---
        out = x + mamba_out * attention
        return out