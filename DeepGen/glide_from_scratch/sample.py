from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image
from transformers import CLIPTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glide_from_scratch.diffusion import DiffusionProcess
from glide_from_scratch.models import GlideModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained GLIDE-style model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)

    model = GlideModel(
        vocab_size=tokenizer.vocab_size,
        max_text_length=config["max_text_length"],
        text_width=config["text_width"],
        text_layers=config["text_layers"],
        text_heads=config["text_heads"],
        model_channels=config["model_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = DiffusionProcess(timesteps=config["timesteps"], device=device)

    prompts = [args.prompt] * args.num_samples
    batch = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=config["max_text_length"],
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if args.negative_prompt.strip():
        negative_batch = tokenizer(
            [args.negative_prompt] * args.num_samples,
            padding="max_length",
            truncation=True,
            max_length=config["max_text_length"],
            return_tensors="pt",
        )
        uncond_input_ids = negative_batch["input_ids"].to(device)
        uncond_attention_mask = negative_batch["attention_mask"].to(device)
    else:
        uncond_input_ids = torch.zeros_like(input_ids)
        uncond_attention_mask = torch.zeros_like(attention_mask)

    if args.sampler == "ddpm":
        images = diffusion.sample_loop(
            model,
            shape=(args.num_samples, 3, config["image_size"], config["image_size"]),
            input_ids=input_ids,
            attention_mask=attention_mask,
            guidance_scale=args.guidance_scale,
            device=device,
            uncond_input_ids=uncond_input_ids,
            uncond_attention_mask=uncond_attention_mask,
        )
    else:
        images = diffusion.ddim_sample_loop(
            model,
            shape=(args.num_samples, 3, config["image_size"], config["image_size"]),
            input_ids=input_ids,
            attention_mask=attention_mask,
            guidance_scale=args.guidance_scale,
            device=device,
            steps=args.steps,
            eta=args.eta,
            uncond_input_ids=uncond_input_ids,
            uncond_attention_mask=uncond_attention_mask,
        )

    images = (images + 1.0) * 0.5
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(images, output_dir / "samples_grid.png", nrow=min(4, args.num_samples))
    for idx, image in enumerate(images):
        save_image(image, output_dir / f"sample_{idx:02d}.png")


if __name__ == "__main__":
    main()
