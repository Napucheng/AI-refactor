from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()


@dataclass
class DiffusionConfig:
    timesteps: int = 1000


class DiffusionProcess:
    def __init__(self, timesteps: int = 1000, device: str | torch.device = "cpu") -> None:
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def training_loss(
        self,
        model,
        x_start: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        caption_dropout: float = 0.1,
    ) -> torch.Tensor:
        batch_size = x_start.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        if caption_dropout > 0:
            keep_mask = (torch.rand(batch_size, device=x_start.device) > caption_dropout).long()
            attention_mask = attention_mask * keep_mask[:, None]
            input_ids = input_ids * keep_mask[:, None]

        pred_noise = model(x_noisy, t, input_ids, attention_mask)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_input_ids: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            pred_noise = model(x, t, input_ids, attention_mask)
        else:
            if uncond_input_ids is None:
                uncond_input_ids = torch.zeros_like(input_ids)
            if uncond_attention_mask is None:
                uncond_attention_mask = torch.zeros_like(attention_mask)
            model_input = torch.cat([x, x], dim=0)
            model_t = torch.cat([t, t], dim=0)
            model_ids = torch.cat([uncond_input_ids, input_ids], dim=0)
            model_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)
            pred_uncond, pred_cond = model(model_input, model_t, model_ids, model_mask).chunk(2, dim=0)
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)

        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape: tuple[int, int, int, int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        guidance_scale: float = 1.0,
        device: str | torch.device = "cpu",
        uncond_input_ids: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            image = self.p_sample(
                model,
                image,
                t,
                input_ids,
                attention_mask,
                guidance_scale=guidance_scale,
                uncond_input_ids=uncond_input_ids,
                uncond_attention_mask=uncond_attention_mask,
            )
        return image.clamp(-1.0, 1.0)

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        shape: tuple[int, int, int, int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        guidance_scale: float = 1.0,
        device: str | torch.device = "cpu",
        steps: int = 100,
        eta: float = 0.0,
        uncond_input_ids: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image = torch.randn(shape, device=device)
        sample_steps = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()

        for idx, t in enumerate(sample_steps):
            batch_t = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            if guidance_scale == 1.0:
                pred_noise = model(image, batch_t, input_ids, attention_mask)
            else:
                if uncond_input_ids is None:
                    uncond_input_ids = torch.zeros_like(input_ids)
                if uncond_attention_mask is None:
                    uncond_attention_mask = torch.zeros_like(attention_mask)
                pred_uncond, pred_cond = model(
                    torch.cat([image, image], dim=0),
                    torch.cat([batch_t, batch_t], dim=0),
                    torch.cat([uncond_input_ids, input_ids], dim=0),
                    torch.cat([uncond_attention_mask, attention_mask], dim=0),
                ).chunk(2, dim=0)
                pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            alpha_bar = self._extract(self.alphas_cumprod, batch_t, image.shape)
            if idx == len(sample_steps) - 1:
                prev_t = torch.zeros_like(batch_t)
            else:
                prev_t = torch.full((shape[0],), int(sample_steps[idx + 1].item()), device=device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alphas_cumprod, prev_t, image.shape)

            x0_pred = (image - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev.clamp_min(1e-8))
            )
            dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp_min(0.0)) * pred_noise
            noise = sigma * torch.randn_like(image)
            image = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + noise

        return image.clamp(-1.0, 1.0)
