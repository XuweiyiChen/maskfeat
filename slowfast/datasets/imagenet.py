# Copyright (c) Facebook, Inc. and its affiliates.

import random
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from PIL import Image
import logging

# Suppress PIL debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)
import sys

sys.path.insert(
    0, "/project/uva_cv_lab/xuweic/SlowFast/env/lib/python3.11/site-packages"
)
from datasets import load_dataset
from torchvision import transforms as transforms_tv
from .build import DATASET_REGISTRY
from .transform import MaskingGenerator, transforms_imagenet_train
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Imagenet(torch.utils.data.Dataset):
    """ImageNet dataset loaded using Hugging Face datasets."""

    def __init__(self, cfg, mode, num_retries=10):
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        if self.mode == "val":
            self.mode = "validation"
        assert mode in [
            "train",
            "validation",
            # "test",
        ], f"Split '{mode}' not supported for ImageNet"
        logger.info(f"Constructing ImageNet {mode} using Hugging Face datasets...")

        # Load dataset using Hugging Face datasets
        self.dataset = load_dataset(
            "/project/uva_cv_lab/dataset/imagenet-100", split=mode
        )

        self.transforms = self._build_transforms()
        self.dummy_output = None

    def _build_transforms(self):
        train_size, test_size = (
            self.cfg.DATA.TRAIN_CROP_SIZE,
            self.cfg.DATA.TEST_CROP_SIZE,
        )
        if self.mode == "train":
            return transforms_imagenet_train(
                img_size=(train_size, train_size),
                color_jitter=self.cfg.AUG.COLOR_JITTER,
                auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION,
                re_prob=self.cfg.AUG.RE_PROB,
                re_mode=self.cfg.AUG.RE_MODE,
                re_count=self.cfg.AUG.RE_COUNT,
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
            )
        else:
            t = []
            if self.cfg.DATA.IN_VAL_CROP_RATIO == 0.0:
                t.append(
                    transforms_tv.Resize(
                        (test_size, test_size),
                        interpolation=transforms_tv.InterpolationMode.BICUBIC,
                    ),
                )
            else:
                size = int((1.0 / self.cfg.DATA.IN_VAL_CROP_RATIO) * test_size)
                t.append(
                    transforms_tv.Resize(
                        size, interpolation=transforms_tv.InterpolationMode.BICUBIC
                    ),
                )
                t.append(transforms_tv.CenterCrop(test_size))
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
            return transforms_tv.Compose(t)

    def _prepare_im_masked(self, image):
        # Apply masking if necessary (for example, for pretraining tasks)
        depth = self.cfg.MASK.PRETRAIN_DEPTH[-1]
        max_mask = self.cfg.AUG.MAX_MASK_PATCHES_PER_BLOCK
        mask_window_size = depth
        num_mask = round(depth * depth * self.cfg.AUG.MASK_RATIO)
        min_mask = num_mask // 5

        mask_generator = MaskingGenerator(
            mask_window_size,
            num_masking_patches=num_mask,
            max_num_patches=max_mask,
            min_num_patches=min_mask,
        )
        mask = mask_generator()
        return [image, torch.Tensor(), mask]

    def __getitem__(self, index):
        for _ in range(self.num_retries):
            try:
                example = self.dataset[index]
                image = example["image"].convert("RGB")
                label = example["label"]
                time = 0  # Dummy value
                meta = {}  # Dummy value
                if self.transforms:
                    image = self.transforms(image)

                if self.cfg.AUG.GEN_MASK_LOADER:
                    return self._prepare_im_masked(image), label, index, time, meta

                return image, label, index, time, meta
            except Exception as e:
                logger.warning(f"Error loading example {index}: {e}")
                index = random.randint(0, len(self.dataset) - 1)

    def __len__(self):
        return len(self.dataset)
