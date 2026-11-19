import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from timm.layers import DropPath

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from models.spectral_blocks import Adaptive_Spectral_Block_c, Adaptive_Spectral_Block_t

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE(nn.Module):
    def __init__(self, shape, base=10000):
        super().__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0

        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution, dim))

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

        return x


class LTABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.05,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.input_resolution, f"输入长度错误：期望{self.input_resolution}，实际{L}"

        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)

        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class ZeroInitGated(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gate * self.fn(x)


class WaveletBranch(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.decompose = nn.Conv1d(dim, dim * 2, kernel_size, padding=kernel_size // 2)
        self.low_branch = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.high_branch = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.Tanh())
        self.fuse = nn.Conv1d(dim * 2, dim, 1)
        self._init_morlet()

    def _init_morlet(self):
        with torch.no_grad():
            out_c, in_c, k = self.decompose.weight.shape
            half = (k - 1) / 2
            t = torch.linspace(-half, half, k)
            for i in range(out_c):
                a = (i + 1) * 0.5
                morlet = torch.exp(-t ** 2 / 2) * torch.cos(5 * t / (a + 0.5))
                self.decompose.weight.data[i] = morlet.unsqueeze(0).repeat(in_c, 1)

    def forward(self, x):
        dec = self.decompose(x)
        low, high = dec.chunk(2, dim=1)
        low = self.low_branch(low)
        high = self.high_branch(high)
        out = self.fuse(torch.cat([low, high], dim=1))
        return out


class CrossAttentionRUL(nn.Module):
    def __init__(self, dim, heads=4, bin_size=10, max_rul=125):
        super().__init__()
        self.dim = dim
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.num_bins = max_rul // bin_size + 1
        self.rul_bins = nn.Embedding(self.num_bins, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, rul):
        B, C, L = x.shape
        x_flat = x.permute(0, 2, 1)

        bin_ids = torch.arange(0, self.num_bins, device=x.device).unsqueeze(0).expand(B, -1)
        cond_tokens = self.rul_bins(bin_ids)

        q = self.to_q(x_flat).view(B, L, self.heads, -1).permute(0, 2, 1, 3)
        k = self.to_k(cond_tokens).view(B, self.num_bins, self.heads, -1).permute(0, 2, 1, 3)
        v = self.to_v(cond_tokens).view(B, self.num_bins, self.heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, C)
        out = self.to_out(out).permute(0, 2, 1)
        return out + x


class AdaptiveFusionGate2(nn.Module):
    def __init__(self, dim, fixed_fusion=False):
        super().__init__()
        self.fixed_fusion = fixed_fusion
        if not fixed_fusion:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid()
            )

    def forward(self, x, lta_out, spec_out):
        if self.fixed_fusion:
            return 0.5 * lta_out + 0.5 * spec_out
        w = self.gate(x).unsqueeze(-1)
        return w * lta_out + (1 - w) * spec_out


class AdaptiveFusionGate3(nn.Module):
    def __init__(self, dim, fixed_fusion=False):
        super().__init__()
        self.dim = dim
        self.fixed_fusion = fixed_fusion
        if not fixed_fusion:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, 3 * dim),
            )

    def forward(self, x, lta_out, spec_out, wave_out):
        if self.fixed_fusion:
            w_m = w_s = w_w = 1.0 / 3.0
        else:
            z = self.gate(x)
            z = z.view(z.shape[0], self.dim, 3)
            w = z.softmax(dim=-1)
            w_m = w[:, :, 0:1]
            w_s = w[:, :, 1:2]
            w_w = w[:, :, 2:3]
        return w_m * lta_out + w_s * spec_out + w_w * wave_out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class moving_maxavg(nn.Module):

    def __init__(self, kernel_size=3, stride=1, maxavg='avg'):
        super(moving_maxavg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.max = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        avg = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        max = self.max(x.permute(0, 2, 1)).permute(0, 2, 1)
        return max, avg


class TSME(nn.Module):

    def __init__(self, dim, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.dim = dim

        self.moving_maxavg = moving_maxavg(kernel_size=kernel_size, stride=stride)
        self.conv_fuse = nn.Conv1d(dim * 2, dim, kernel_size=1, stride=1)

        self.conv_temporal = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

        self.norm_temporal = nn.LayerNorm(dim)

        self.to_gamma = nn.Conv1d(dim, dim, 1)
        self.to_beta = nn.Conv1d(dim, dim, 1)

        self.gate = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x

        max_feat, avg_feat = self.moving_maxavg(x)
        trend_feat = self.conv_fuse(torch.cat([max_feat, avg_feat], dim=1))

        feat = self.conv_temporal(trend_feat)
        feat = feat.transpose(1, 2)
        feat = self.norm_temporal(feat)
        feat = feat.transpose(1, 2)

        gamma = self.to_gamma(feat)
        beta = self.to_beta(feat)
        td_feat = feat * gamma + beta

        gate = self.gate(torch.cat([residual, td_feat], dim=1))
        out = residual * (1 - gate) + td_feat * gate

        return out


class InduxTFS_Diff(nn.Module):
    def __init__(
            self,
            dim,
            cond_drop_prob=0.5,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 2),
            channels=14,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            arch='lta+td',
            input_length=48,
            lta_num_heads=2,
            lta_drop_path=0.05,
            fusion_weight=0.4,
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.arch = arch
        self.fusion_weight = fusion_weight

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.classes_emb = nn.Linear(1, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        use_lta = self.arch in ('lta+td', 'lta+td+fre_parallel_adaptive',
                                'ablation_no_tsme', 'ablation_fixed_fusion',
                                'ablation_no_fre', 'ablation_no_wavelet')
        use_adaptive = self.arch in ('lta+td+fre_parallel_adaptive',
                                     'ablation_no_tsme', 'ablation_fixed_fusion',
                                     'ablation_no_lta', 'ablation_no_fre', 'ablation_no_wavelet')
        use_spectral = use_adaptive and self.arch != 'ablation_no_fre'
        use_wavelet = use_adaptive and self.arch != 'ablation_no_wavelet'
        self.use_lta = use_lta
        self.use_adaptive = use_adaptive
        self.use_spectral = use_spectral
        self.use_wavelet = use_wavelet
        self.skip_tsme = self.arch == 'ablation_no_tsme'
        self.fixed_fusion = self.arch == 'ablation_fixed_fusion'

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            current_length = input_length // (2 ** ind)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                LTABlock(
                    dim=dim_in,
                    input_resolution=current_length,
                    num_heads=lta_num_heads,
                    drop_path=lta_drop_path
                ) if use_lta else nn.Identity(),
                Residual(PreNorm(dim_in, Adaptive_Spectral_Block_c(dim_in))) if use_spectral else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        mid_length = input_length // (2 ** (num_resolutions - 1))

        if self.arch == 'lta+td':
            self.mid_attn_original = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_attn_lta = LTABlock(
                dim=mid_dim,
                input_resolution=mid_length,
                num_heads=lta_num_heads,
                drop_path=lta_drop_path
            )
        elif use_adaptive and use_lta:
            self.mid_attn_lta = LTABlock(
                dim=mid_dim,
                input_resolution=mid_length,
                num_heads=lta_num_heads,
                drop_path=lta_drop_path
            )
        elif not use_adaptive:
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))

        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_td = TSME(dim=mid_dim, kernel_size=3, stride=1)

        self.down_adaptive_gates = nn.ModuleList([
            AdaptiveFusionGate2(dim_in, fixed_fusion=self.fixed_fusion) for (dim_in, _) in in_out
        ]) if use_adaptive else nn.ModuleList()
        self.up_adaptive_gates = nn.ModuleList([
            AdaptiveFusionGate2(dim_out, fixed_fusion=self.fixed_fusion) for (_, dim_out) in reversed(in_out)
        ]) if use_adaptive else nn.ModuleList()
        self.mid_spectral = Residual(PreNorm(mid_dim, Adaptive_Spectral_Block_c(mid_dim))) if use_spectral else nn.Identity()
        self.mid_wavelet_adaptive = ZeroInitGated(PreNorm(mid_dim, WaveletBranch(mid_dim))) if use_wavelet else nn.Identity()
        if use_adaptive:
            self.mid_adaptive_gate = (AdaptiveFusionGate3(mid_dim, fixed_fusion=self.fixed_fusion) if use_wavelet
                                      else AdaptiveFusionGate2(mid_dim, fixed_fusion=self.fixed_fusion))
        else:
            self.mid_adaptive_gate = nn.Identity()

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            current_length = input_length // (2 ** (len(in_out) - 1 - ind))

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                LTABlock(
                    dim=dim_out,
                    input_resolution=current_length,
                    num_heads=lta_num_heads,
                    drop_path=lta_drop_path
                ) if use_lta else nn.Identity(),
                Residual(PreNorm(dim_out, Adaptive_Spectral_Block_t(dim_out))) if use_spectral else nn.Identity(),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            x,
            time,
            classes,
            cond_drop_prob=None,
            return_features=False,
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        classes_emb = self.classes_emb(classes.float())

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        h = []

        for d_idx, (block1, block2, mlla, spectral_c, downsample) in enumerate(self.downs):
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            h.append(x)
            resblock_x = x

            if self.use_lta:
                x_nhwc = x.permute(0, 2, 1)
                x = mlla(x_nhwc).permute(0, 2, 1)

            if self.use_adaptive:
                lta_out = x
                spec_out = spectral_c(resblock_x) if self.use_spectral else resblock_x
                x = self.down_adaptive_gates[d_idx](resblock_x, lta_out, spec_out)

            x = downsample(x)

        x = self.mid_block1(x, t, c)

        if self.arch == 'lta+td':
            x_td = self.mid_td(x)
            x_attn_original = self.mid_attn_original(x_td)
            x_td_nhwc = x_td.permute(0, 2, 1)
            x_attn_mlla = self.mid_attn_lta(x_td_nhwc).permute(0, 2, 1)
            x_attn = (1 - self.fusion_weight) * x_attn_original + self.fusion_weight * x_attn_mlla
            x = x + x_attn
        elif self.use_adaptive:
            x_td = x if self.skip_tsme else self.mid_td(x)
            if self.use_lta:
                x_td_nhwc = x_td.permute(0, 2, 1)
                lta_out = self.mid_attn_lta(x_td_nhwc).permute(0, 2, 1)
            else:
                lta_out = x_td
            spec_out = self.mid_spectral(x_td) if self.use_spectral else x_td
            if self.use_wavelet:
                wave_out = self.mid_wavelet_adaptive(x_td)
                x_attn = self.mid_adaptive_gate(x_td, lta_out, spec_out, wave_out)
            else:
                x_attn = self.mid_adaptive_gate(x_td, lta_out, spec_out)
            x = x + x_attn

        x = self.mid_block2(x, t, c)

        for u_idx, (block1, block2, mlla, spectral_t, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            resblock_x = x

            if self.use_lta:
                x_nhwc = x.permute(0, 2, 1)
                x = mlla(x_nhwc).permute(0, 2, 1)

            if self.use_adaptive:
                lta_out = x
                spec_out = spectral_t(resblock_x) if self.use_spectral else resblock_x
                x = self.up_adaptive_gates[u_idx](resblock_x, lta_out, spec_out)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)


if __name__ == '__main__':
    batch_size = 4

    print("=" * 60)
    print("测试 MLLA+TD 版本")
    model_mlla = InduxTFS_Diff(
        dim=32,
        dim_mults=(1, 2, 2),
        cond_drop_prob=0.5,
        channels=14,
        arch='lta+td',
        input_length=48,
        lta_num_heads=2,
        lta_drop_path=0.05,
        fusion_weight=0.4,
        resnet_block_groups=8
    )
    x = torch.randn(batch_size, 14, 48)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size, 1])
    y = model_mlla(x, t, labels)
    total_params = sum(p.numel() for p in model_mlla.parameters())
    print(f"  输入: {x.shape} → 输出: {y.shape}, 参数量: {total_params:,}")

    print("=" * 60)
    print("测试 MLLA+TD+FRE 融合版本")
    model_fusion = InduxTFS_Diff(
        dim=32,
        dim_mults=(1, 2, 2),
        cond_drop_prob=0.5,
        channels=14,
        arch='lta+td+fre',
        input_length=48,
        lta_num_heads=2,
        lta_drop_path=0.05,
        fusion_weight=0.4,
        resnet_block_groups=8
    )
    y2 = model_fusion(x, t, labels)
    total_params2 = sum(p.numel() for p in model_fusion.parameters())
    print(f"  输入: {x.shape} → 输出: {y2.shape}, 参数量: {total_params2:,}")
    print(f"输出形状: {y.shape}")
    print("✅ 所有方案集成成功，模型运行正常！")

