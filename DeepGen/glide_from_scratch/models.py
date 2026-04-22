from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 64,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Parameter(torch.randn(1, max_length, width) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln = nn.LayerNorm(width)
        self.pool = nn.Linear(width, width)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask.unsqueeze(-1).float()
        x = self.token_embedding(input_ids) + self.position_embedding[:, : input_ids.size(1)]
        x = x * mask
        key_padding_mask = attention_mask == 0
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)
        x = x * mask
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.pool(pooled)
        return x, pooled


class CrossAttention2d(nn.Module):
    def __init__(self, channels: int, context_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x).view(b, c, h * w).transpose(1, 2)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        head_dim = c // self.heads
        q = q.view(b, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, -1e4)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = self.proj(out).transpose(1, 2).view(b, c, h, w)
        return residual + out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class UNetConditionModel(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        model_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        text_width: int = 256,
        time_width: int = 256,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_width),
            nn.SiLU(),
            nn.Linear(time_width, time_width),
        )
        self.cond_proj = nn.Linear(text_width, time_width)
        self.input_conv = nn.Conv2d(image_channels, model_channels, kernel_size=3, padding=1)

        downs = []
        channels = model_channels
        skip_channels = [channels]
        for level, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            downs.append(ResBlock(channels, out_channels, time_width))
            channels = out_channels
            downs.append(CrossAttention2d(channels, text_width))
            skip_channels.append(channels)
            if level != len(channel_mults) - 1:
                downs.append(Downsample(channels))
                skip_channels.append(channels)
        self.downs = nn.ModuleList(downs)

        self.mid_block1 = ResBlock(channels, channels, time_width)
        self.mid_attn = CrossAttention2d(channels, text_width)
        self.mid_block2 = ResBlock(channels, channels, time_width)

        ups = []
        rev_mults = list(reversed(channel_mults))
        rev_skips = list(reversed(skip_channels))
        for level, mult in enumerate(rev_mults):
            out_channels = model_channels * mult
            ups.append(ResBlock(channels + rev_skips.pop(0), out_channels, time_width))
            channels = out_channels
            ups.append(CrossAttention2d(channels, text_width))
            if level != len(rev_mults) - 1:
                ups.append(Upsample(channels))
                ups.append(ResBlock(channels + rev_skips.pop(0), channels, time_width))
        self.ups = nn.ModuleList(ups)

        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_tokens: torch.Tensor,
        text_pooled: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_mlp(t_emb) + self.cond_proj(text_pooled)

        h = self.input_conv(x)
        skips = [h]
        for module in self.downs:
            if isinstance(module, ResBlock):
                h = module(h, emb)
                skips.append(h)
            elif isinstance(module, CrossAttention2d):
                h = module(h, text_tokens, attention_mask)
            else:
                h = module(h)
                skips.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h, text_tokens, attention_mask)
        h = self.mid_block2(h, emb)

        for module in self.ups:
            if isinstance(module, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, emb)
            elif isinstance(module, CrossAttention2d):
                h = module(h, text_tokens, attention_mask)
            else:
                h = module(h)

        return self.out(h)


class GlideModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_text_length: int = 64,
        text_width: int = 256,
        text_layers: int = 4,
        text_heads: int = 8,
        model_channels: int = 128,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_text_length,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
        )
        self.unet = UNetConditionModel(
            image_channels=3,
            model_channels=model_channels,
            text_width=text_width,
            time_width=text_width,
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_tokens, text_pooled = self.text_encoder(input_ids, attention_mask)
        return self.unet(noisy_images, timesteps, text_tokens, text_pooled, attention_mask)
