from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedCellDataset(Dataset):
    def __init__(self, records: Sequence, bf_dir: Path, fl_dir: Path, transform: transforms.Compose, is_test: bool = False) -> None:
        self.records = records.reset_index(drop=True) if hasattr(records, "reset_index") and hasattr(records, "iloc") else list(records)
        self._uses_dataframe_rows = hasattr(self.records, "iloc")
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records.iloc[index] if self._uses_dataframe_rows else self.records[index]
        image_name = record["Name"] if hasattr(record, "__getitem__") else record.Name

        bf_image = self._load_image(self.bf_dir / image_name)
        fl_image = self._load_image(self.fl_dir / image_name)

        bf_tensor, fl_tensor = self._apply_transform(bf_image, fl_image)

        if self.is_test:
            return bf_tensor, fl_tensor, image_name

        label = record["Diagnosis"] if hasattr(record, "__getitem__") else record.Diagnosis
        return bf_tensor, fl_tensor, torch.tensor(float(label), dtype=torch.float32)

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _apply_transform(self, bf_image: Image.Image, fl_image: Image.Image):
        if self.transform is None:
            to_tensor = transforms.ToTensor()
            return to_tensor(bf_image), to_tensor(fl_image)

        try:
            transformed = self.transform(bf_image, fl_image)
        except TypeError:
            transformed = None

        if transformed is not None:
            return transformed

        return self.transform(bf_image), self.transform(fl_image)
