from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class Sample:
    image: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    caption: str


def build_image_transform(image_size: int, random_flip: bool = True) -> Callable[[Image.Image], torch.Tensor]:
    transform_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ]
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_list)


class CaptionedImageDataset(Dataset[Sample]):
    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path,
        tokenizer,
        image_size: int = 64,
        max_length: int = 64,
        random_flip: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = build_image_transform(image_size=image_size, random_flip=random_flip)
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, str]]:
        with self.metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        if not records:
            raise ValueError(f"No records found in {self.metadata_path}")
        if "filename" not in records[0] or "caption" not in records[0]:
            raise ValueError("metadata.csv must contain filename and caption columns")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        image_path = self.data_root / "images" / record["filename"]
        caption = record["caption"].strip()
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        encoded = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return Sample(
            image=image_tensor,
            input_ids=encoded["input_ids"].squeeze(0),
            attention_mask=encoded["attention_mask"].squeeze(0),
            caption=caption,
        )


def collate_samples(samples: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "images": torch.stack([sample.image for sample in samples], dim=0),
        "input_ids": torch.stack([sample.input_ids for sample in samples], dim=0),
        "attention_mask": torch.stack([sample.attention_mask for sample in samples], dim=0),
        "captions": [sample.caption for sample in samples],
    }
