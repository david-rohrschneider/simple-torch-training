import os
import hashlib
import time

import torch
from torchvision.datasets import ImageFolder


class DataLoader:
    """Wrapper class for generating multiple dataloaders from config"""

    def __init__(self, device):
        """Constructor for DataLoader"""

        self.train_dataset = None
        self.__device = device

    @staticmethod
    def __get_cache_path(dir_path: str):
        hashed_path = hashlib.sha1(dir_path.encode()).hexdigest()
        cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder",
                                  hashed_path[:10] + ".pt")
        cache_path = os.path.expanduser(cache_path)
        return cache_path

    def cache_dataset(self, dir_path: str, cache_path: str):
        """Cache dataset"""

        os.mkdir(os.path.dirname(cache_path))
        utils.save_on_master((self.train_dataset, dir_path), cache_path)

    def load_train_dataset(self, train_path: str, cached: bool,
                           train_crop_size: int, auto_augment_policy: str, random_erase_prob: float,
                           ra_magnitude: int, augmix_severity: int, interpolation: str):
        """Load train dataset"""

        cache_path = self.__get_cache_path(train_path)
        if cached and os.path.exists(cache_path):
            start_time = time.time()
            self.train_dataset, _ = torch.load(cache_path)
            end_time = time.time()
            msg = f"Loaded cached dataset from {cache_path} in {end_time-start_time}s!"
        else:
            start_time = time.time()
            self.train_dataset = ImageFolder(
                train_path,
                presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                ),
            )
            end_time = time.time()
            msg = f"Loaded dataset from {train_path} in {end_time-start_time}s!"
        return msg, cache_path
