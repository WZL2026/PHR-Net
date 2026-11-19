import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- mamba 环境兼容（必须在 import mamba_ssm 之前）----
# 集群上的 mamba_ssm 依赖较新 triton 的 set_allocator，且包 __init__ 会连带 import
# transformers（仅 LM 头用得到）。这里先装好兼容垫片，否则任何直接 import 本模型的
# 代码（例如 test_comparison.py 有自己的 get_model）都会
#   AttributeError: module 'triton' has no attribute 'set_allocator'
# 而训练/测试流程表现不一致（训练能过、测试挂）。
try:
    from baselines_ext.mamba_fallback import install_mamba_fallback as _install_mamba_fallback
    _install_mamba_fallback()
except Exception:  # noqa: BLE001  垫片失败不阻塞（真实 mamba_ssm 可用时本来也不需要）
    pass

from mamba_ssm import Mamba

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_fwd = self.mamba(x_norm)
        x_rev = self.mamba(x_norm.flip([1])).flip([1])
        x_mamba = (x_fwd + x_rev) / 2.0
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, downsample=True):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
            )

        self.mamba_layers = nn.Sequential(
            *[MambaLayer(dim=out_channels) for _ in range(depth)]
        )

        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(out_channels, out_channels * 4)

    def forward(self, x):
        if self.downsample:
            x = self.downsample_layer(x)

        x = self.mamba_layers(x)
        x = self.norm(x)
        x = self.mlp(x)

        return x

def downsample_layer(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
    )

class UnetrBasicBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, norm_name="instance", res_block=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2)
        self.norm1 = nn.InstanceNorm2d(out_channels) if norm_name == "instance" else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size//2)
        self.norm2 = nn.InstanceNorm2d(out_channels) if norm_name == "instance" else nn.BatchNorm2d(out_channels)
        self.res_block = res_block
        if res_block and in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.res_block:
            x += residual
        x = self.relu(x)
        return x

class UnetrUpBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size=3, upsample_kernel_size=2, norm_name="instance", res_block=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, upsample_kernel_size, upsample_kernel_size)
        self.conv = UnetrBasicBlock(spatial_dims, out_channels * 2, out_channels, kernel_size, 1, norm_name, res_block)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class MambaVesselNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, feature_dims=None, norm_name="instance", spatial_dims=2, res_block=True):
        super().__init__()

        if feature_dims is None:
            feature_dims = [48, 96, 192, 384, 768]

        # Encoder
        self.enc_conv1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=feature_dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample1 = downsample_layer(feature_dims[0], feature_dims[0])

        self.enc_conv2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[0],
            out_channels=feature_dims[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample2 = downsample_layer(feature_dims[1], feature_dims[1])

        self.enc_conv3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[1],
            out_channels=feature_dims[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample3 = downsample_layer(feature_dims[2], feature_dims[2])

        self.enc_conv4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[2],
            out_channels=feature_dims[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.downsample4 = downsample_layer(feature_dims[3], feature_dims[3])

        self.enc_conv5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[3],
            out_channels=feature_dims[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Encoder Mamba Blocks
        self.enc_mamba1 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.enc_mamba2 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.enc_mamba3 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.enc_mamba4 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)

        # Decoder Mamba Blocks
        self.dec_mamba1 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.dec_mamba2 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.dec_mamba3 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)
        self.dec_mamba4 = MambaBlock(in_channels=feature_dims[4], out_channels=feature_dims[4],
                                     depth=2, downsample=False)

        # Decoder Conv Layers
        self.dec_conv5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[4],
            out_channels=feature_dims[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[3],
            out_channels=feature_dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[2],
            out_channels=feature_dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[1],
            out_channels=feature_dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.dec_conv1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_dims[0],
            out_channels=feature_dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # Final output layer
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_dims[0], out_channels=out_ch)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc1_down = self.downsample1(enc1)

        enc2 = self.enc_conv2(enc1_down)
        enc2_down = self.downsample2(enc2)

        enc3 = self.enc_conv3(enc2_down)
        enc3_down = self.downsample3(enc3)

        enc4 = self.enc_conv4(enc3_down)
        enc4_down = self.downsample4(enc4)

        enc5 = self.enc_conv5(enc4_down)

        x = self.enc_mamba1(enc5)
        x = self.enc_mamba2(x)
        x = self.enc_mamba3(x)
        x = self.enc_mamba4(x)

        x = self.dec_mamba1(x)
        x = self.dec_mamba2(x)
        x = self.dec_mamba3(x)
        x = self.dec_mamba4(x)

        dec5 = self.dec_conv5(x, enc4)
        dec4 = self.dec_conv4(dec5, enc3)
        dec3 = self.dec_conv3(dec4, enc2)
        dec2 = self.dec_conv2(dec3, enc1)

        x = self.dec_conv1(dec2)
        out = self.out(x)

        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaVesselNet(in_ch=3, out_ch=1).to(device)
    input = torch.randn(1, 3, 512, 512).to(device)
    output = model(input)
    print(f'Input shape: {input.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')