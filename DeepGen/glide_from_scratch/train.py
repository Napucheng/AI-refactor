from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import CLIPTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glide_from_scratch.dataset import CaptionedImageDataset, collate_samples
from glide_from_scratch.diffusion import DiffusionProcess
from glide_from_scratch.models import GlideModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GLIDE-style text-to-image diffusion model from scratch.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--text-width", type=int, default=256)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--text-heads", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-prompts", nargs="*", default=None)
    parser.add_argument("--sample-guidance-scale", type=float, default=5.0)
    parser.add_argument("--sample-steps", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: GlideModel, optimizer: AdamW, epoch: int, args: argparse.Namespace) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args),
    }
    torch.save(payload, path)


@torch.no_grad()
def sample_preview(
    model: GlideModel,
    tokenizer: CLIPTokenizer,
    diffusion: DiffusionProcess,
    prompts: list[str],
    output_dir: Path,
    image_size: int,
    max_text_length: int,
    guidance_scale: float,
    steps: int,
    device: torch.device,
    epoch: int,
) -> None:
    model.eval()
    batch = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    images = diffusion.ddim_sample_loop(
        model,
        shape=(len(prompts), 3, image_size, image_size),
        input_ids=input_ids,
        attention_mask=attention_mask,
        guidance_scale=guidance_scale,
        device=device,
        steps=steps,
    )
    images = (images + 1.0) * 0.5
    save_image(images, output_dir / f"preview_epoch_{epoch:04d}.png", nrow=min(4, len(prompts)))
    model.train()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
    dataset = CaptionedImageDataset(
        data_root=args.data_root,
        metadata_path=args.metadata,
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_length=args.max_text_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_samples,
    )

    model = GlideModel(
        vocab_size=tokenizer.vocab_size,
        max_text_length=args.max_text_length,
        text_width=args.text_width,
        text_layers=args.text_layers,
        text_heads=args.text_heads,
        model_channels=args.model_channels,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    diffusion = DiffusionProcess(timesteps=args.timesteps, device=device)

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(
                model=model,
                x_start=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                caption_dropout=args.caption_dropout,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        save_checkpoint(output_dir / "model_latest.pt", model, optimizer, epoch, args)
        if epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"model_epoch_{epoch:04d}.pt", model, optimizer, epoch, args)

        if args.sample_prompts:
            sample_preview(
                model=model,
                tokenizer=tokenizer,
                diffusion=diffusion,
                prompts=args.sample_prompts,
                output_dir=output_dir,
                image_size=args.image_size,
                max_text_length=args.max_text_length,
                guidance_scale=args.sample_guidance_scale,
                steps=args.sample_steps,
                device=device,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
