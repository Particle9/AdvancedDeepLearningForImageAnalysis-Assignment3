from __future__ import annotations

import random

from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BasePairTransform:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, bf_image, fl_image):
        bf_image = TF.resize(bf_image, [self.image_size, self.image_size])
        fl_image = TF.resize(fl_image, [self.image_size, self.image_size])
        bf_tensor = self.normalize(TF.to_tensor(bf_image))
        fl_tensor = self.normalize(TF.to_tensor(fl_image))
        return bf_tensor, fl_tensor


class AugmentedPairTransform:
    def __init__(
        self,
        image_size: int = 224,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, bf_image, fl_image):
        bf_image = TF.resize(bf_image, [self.image_size, self.image_size])
        fl_image = TF.resize(fl_image, [self.image_size, self.image_size])

        if random.random() < self.hflip_prob:
            bf_image = TF.hflip(bf_image)
            fl_image = TF.hflip(fl_image)

        if random.random() < self.vflip_prob:
            bf_image = TF.vflip(bf_image)
            fl_image = TF.vflip(fl_image)

        if random.random() < self.rotate_prob:
            angle = random.choice([90, 180, 270])
            bf_image = TF.rotate(bf_image, angle)
            fl_image = TF.rotate(fl_image, angle)

        bf_tensor = self.normalize(TF.to_tensor(bf_image))
        fl_tensor = self.normalize(TF.to_tensor(fl_image))
        return bf_tensor, fl_tensor
