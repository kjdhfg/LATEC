from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import random
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
from torchvision import datasets

from src.utils.download_url import *


class RESISC45DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/datasets/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes=45,
        resize=224,
        resize_mode="bilinear",
        modality: str = "image",
        weights_resnet="IMAGENET1K_V1",
        weights_effnet="IMAGENET1K_V1",
        weights_vit="IMAGENET1K_V1",
    ):
        super().__init__()
        self.__name__ = "resisc45"

        if not os.path.exists(data_dir + "/NWPU-RESISC45"):
            raise ValueError(
                "Manual download required: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html. Please relocate the 'NWPU-RESISC45' folder to the 'dataset' directory."
            )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.data = datasets.ImageFolder(
            data_dir + "/NWPU-RESISC45/", transform=self.transforms
        )

        self.g = torch.Generator()
        self.g.manual_seed(0)

    @property
    def num_classes(self):
        return 4

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def dataloader(self):
        return DataLoader(
            dataset=self.data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )


if __name__ == "__main__":
    _ = RESISC45DataModule()
