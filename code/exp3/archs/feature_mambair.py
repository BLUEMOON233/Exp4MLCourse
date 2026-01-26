import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    raise ImportError("请确保已正确安装 mamba_ssm 环境。")

# =============================================================================
# 辅助模块：VGG 特征提取器（用于 Loss 计算）
# =============================================================================
class VGGPerceptualLossExtractor(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        all_features = vgg.features
        self.block1 = nn.Sequential(*list(all_features.children())[:10]) # -> 128 ch (conv2_2)
        self.block2 = nn.Sequential(*list(all_features.children())[10:17]) # -> 256 ch (conv3_3)
        self.block3 = nn.Sequential(*list(all_features.children())[17:24]) # -> 512 ch (conv4_3)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return f1, f2, f3

# =============================================================================
# MambaIR 核心组件 (精简复刻版)
# =============================================================================

class PatchEmbed(nn.Module):
    """ 2D Image to 1D Token Sequence (with optional Norm) """
    def __init__(self, img_size=224, patch_size=1, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        # patch_size=1 意味着不进行下采样，只是 Reshape
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Flatten: [B, C, H*W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    """ 1D Token Sequence to 2D Image """
    def __init__(self, img_size=224, patch_size=1, in_chans=64, embed_dim=64, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # x: [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.attention(x)

class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )
    def forward(self, x):
        return self.cab(x)

class SS2D(nn.Module):
    """ Select Scan 2D: MambaIR 核心注意力模块 """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2., dropout=0., device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=True, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
            bias=True, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=True, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            nn.init.constant_(dt_proj.weight, dt_init_std)
        
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, merge=True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
        ).view(B, K, -1, L)
        
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0, norm_layer=nn.LayerNorm, d_state=16, expand=2.):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input):
        # input: [B, H*W, C] (Expects tokens) - Wait, SS2D/CAB logic above handled shapes differently?
        # Let's align: SS2D expects [B, H, W, C].
        B, H, W, C = input.shape
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        # CAB expects [B, C, H, W]
        x_cab = self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        x_cab = self.conv_blk(x_cab)
        x_cab = x_cab.permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + self.drop_path(x_cab)
        return x

class ResidualGroup(nn.Module):
    def __init__(self, dim, depth, d_state=16, mlp_ratio=2., norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, d_state=d_state, expand=mlp_ratio, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        # x is [B, L, C], we need to reshape for blocks if they expect [B,H,W,C]
        H, W = x_size
        B, L, C = x.shape
        x_spatial = x.view(B, H, W, C)
        
        for blk in self.blocks:
            x_spatial = blk(x_spatial)
            
        # Back to [B, C, H, W] for Conv
        x_conv = x_spatial.permute(0, 3, 1, 2).contiguous()
        x_conv = self.conv(x_conv)
        # Back to [B, L, C]
        x_out = x_conv.flatten(2).transpose(1, 2)
        return x_out + x

class GlobalCorrection(nn.Module):
    """
    可学习的全局特征校正 (Global Scale/Shift)
    功能: 专门处理亮度(Brightness)/对比度(Contrast)等全图一致的统计量偏移。
    原理: x_out = x * (1 + scale) + shift
    复杂度: 极低 (Global AvgPool + MLP)
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim * 2, bias=True)
        )
        # 初始化处理：让初始状态接近 Identity，避免训练初期震荡
        # scale 部分输出如果接近 0，则 (1+scale) 接近 1
        # shift 部分输出接近 0
        nn.init.constant_(self.fc[-1].weight, 0)
        nn.init.constant_(self.fc[-1].bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        stats = self.avg_pool(x).view(b, c)
        params = self.fc(stats).view(b, c * 2, 1, 1)
        scale, shift = params.chunk(2, dim=1)
        return x * (1.0 + scale) + shift

class MambaFeatureEnhancer(nn.Module):
    def __init__(self, in_channels=64, img_channels=3, dims=[128, 64, 32], blocks_per_group=4):
        super().__init__()
        self.dims = dims
        
        # 0. Global Correction Module (Pre-processing)
        # 在进入复杂的 Mamba 纹理修复前，先用极轻量的模块修正全局亮度/对比度
        self.global_correct = GlobalCorrection(in_channels)
        
        # FDM-inspired Cascade Structure (Eq. 4 in Paper)
        # Branch 1 (Center / G1) - Takes Raw Input
        # In: (Feature + Image) -> Out: 128
        self.b1_conv = nn.Conv2d(in_channels + img_channels, dims[0], 3, 1, 1)
        self.b1_group = ResidualGroup(dim=dims[0], depth=blocks_per_group)
        
        # Branch 2 (Surround / G2') - Takes Branch 1 Output
        # In: 128 -> Out: 64
        self.b2_conv = nn.Conv2d(dims[0], dims[1], 3, 1, 1)
        self.b2_group = ResidualGroup(dim=dims[1], depth=blocks_per_group)
        
        # Branch 3 (Margin / G3') - Takes Branch 2 Output
        # In: 64 -> Out: 32
        self.b3_conv = nn.Conv2d(dims[1], dims[2], 3, 1, 1)
        self.b3_group = ResidualGroup(dim=dims[2], depth=blocks_per_group)
        
        # Fusion & Dimension Matching
        total_dim = sum(dims)
        self.fusion = nn.Conv2d(total_dim, in_channels, 1, 1, 0)
        
    def forward(self, x, img):
        """
        Args:
            x: Degraded Features [B, C, H, W]
            img: Degraded Image [B, 3, H_img, W_img]
        """
        B, C, H, W = x.shape
        
        # 0. Apply Global Correction first
        # 修复亮度/对比度的统计偏移，让后续网络专注于纹理恢复
        x = self.global_correct(x)

        
        # 1. Downsample image to match feature spatial dimensions
        img_down = F.interpolate(img, size=(H, W), mode='bilinear', align_corners=False)
        
        # 2. Concat Features + Image
        inp = torch.cat([x, img_down], dim=1)
        
        # 3. Process Branches in Cascade (Sequential)
        
        # Branch 1 (Input -> x1)
        feat1 = self.b1_conv(inp)
        feat1_flat = feat1.flatten(2).transpose(1, 2)
        x1_flat = self.b1_group(feat1_flat, (H, W))
        x1 = x1_flat.transpose(1, 2).reshape(B, self.dims[0], H, W)
        
        # Branch 2 (x1 -> x2)
        feat2 = self.b2_conv(x1) # Cascade connection
        feat2_flat = feat2.flatten(2).transpose(1, 2)
        x2_flat = self.b2_group(feat2_flat, (H, W))
        x2 = x2_flat.transpose(1, 2).reshape(B, self.dims[1], H, W)
        
        # Branch 3 (x2 -> x3)
        feat3 = self.b3_conv(x2) # Cascade connection
        feat3_flat = feat3.flatten(2).transpose(1, 2)
        x3_flat = self.b3_group(feat3_flat, (H, W))
        x3 = x3_flat.transpose(1, 2).reshape(B, self.dims[2], H, W)
        
        # 4. Fusion
        # Concatenate outputs from all stages (simulating summation of terms in Eq. 4)
        concat = torch.cat([x1, x2, x3], dim=1)
        residual = self.fusion(concat)
        
        # 5. Residual Addition
        return x + residual
