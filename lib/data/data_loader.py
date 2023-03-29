import os
import hashlib

import torch
from torchvision.transforms import autoaugment, transforms

import albumentations as at
import cv2

from lib.config_loaders.data_config import DataConfig
from .datasets import get_custom_dataset_class

from ..helpers import decorators, enums, constants


class DataLoader:
    """Wrapper class for generating multiple dataloaders from config"""

    def __init__(self, custom: bool, device):
        """Constructor for DataLoader"""

        self.config: DataConfig = DataConfig(custom=custom)
        self.transforms_train = self.__get_transforms_train()
        self.transforms_eval = self.__get_transforms_eval()
        self.train_dataset = None
        self.val_dataset = None
        self.__device = device

    @staticmethod
    def get_cache_path(dir_path: str):
        hashed_path = hashlib.sha1(dir_path.encode()).hexdigest()
        cache_path = os.path.join("~", enums.CacheDirNames.ROOT.value,
                                  enums.CacheDirNames.DATASETS.value,
                                  f"{hashed_path[:10]}.pt")
        cache_path = os.path.expanduser(cache_path)
        return cache_path

    @decorators.stop_time
    def load_cached_dataset(self, cache_path: str, is_train: bool = True):
        if is_train:
            self.train_dataset, _ = torch.load(cache_path)
        else:
            self.val_dataset, _ = torch.load(cache_path)

    def cache_dataset(self, dir_path: str, cache_path: str):
        """Cache dataset"""
        utils.save_on_master((self.train_dataset, dir_path), cache_path)

    def __get_transforms_classification_train(self):
        trans = []
        interpolation = transforms.InterpolationMode(self.config.interpolation_mode)
        if self.config.crop and self.config.train_crop_size:
            trans.append(transforms.RandomResizedCrop(self.config.train_crop_size,
                                                      interpolation=interpolation))
        if self.config.hor_flipping and self.config.h_flip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(self.config.h_flip_prob))
        if self.config.auto_augment and self.config.auto_augment_policy:
            if self.config.auto_augment_policy == "randomaugment":
                trans.append(
                    autoaugment.RandAugment(interpolation=interpolation,
                                            magnitude=self.config.magnitude))
            elif self.config.auto_augment_policy == "trivialaugmentwide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif self.config.auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(interpolation=interpolation, severity=self.config.severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(self.config.auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        if self.config.normalize and self.config.norm_mean and self.config.norm_std:
            trans.append(transforms.Normalize(self.config.norm_mean, self.config.norm_std))
        if self.config.random_erase_prob and self.config.random_erase_prob > 0:
            trans.append(transforms.RandomErasing(self.config.random_erase_prob))

        trans.extend([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
        return transforms.Compose(transforms=trans)

    def __get_transforms_detection_train(self):
        trans = []
        interpolation = 1
        if self.config.interpolation_mode in constants.CV2_INTERPOLATION_MODES:
            interpolation = constants.CV2_INTERPOLATION_MODES[self.config.interpolation_mode]
        if self.config.crop and self.config.train_crop_size:
            if len(self.config.train_crop_size) == 1:
                height = width = self.config.train_crop_size[0]
            else:
                height = self.config.train_crop_size[0]
                width = self.config.train_crop_size[1]
            trans.append(at.RandomResizedCrop(height=height,
                                              width=width,
                                              interpolation=interpolation))
        if self.config.hor_flipping and self.config.h_flip_prob > 0:
            trans.append(at.HorizontalFlip(self.config.h_flip_prob))
        if self.config.auto_augment:
            norm_severity = self.config.severity / 10

        if self.config.normalize and self.config.norm_mean and self.config.norm_std:
            trans.append(transforms.Normalize(self.config.norm_mean, self.config.norm_std))
        if self.config.random_erase_prob and self.config.random_erase_prob > 0:
            trans.append(transforms.RandomErasing(self.config.random_erase_prob))

        trans.extend([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
        return transforms.Compose(transforms=trans)

    def __get_transforms_classification_eval(self):
        trans = []
        interpolation = transforms.InterpolationMode(self.config.interpolation_mode)
        if self.config.resize and self.config.eval_resize:
            trans.append(transforms.Resize(self.config.eval_resize, interpolation=interpolation))
        if self.config.crop and self.config.eval_crop_size:
            trans.append(transforms.CenterCrop(self.config.eval_crop_size))
        if self.config.normalize and self.config.norm_mean and self.config.norm_std:
            trans.append(transforms.Normalize(mean=self.config.norm_mean, std=self.config.norm_std))

        trans.extend([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
        return transforms.Compose(transforms=trans)

    @decorators.stop_time
    def load_dataset(self, split_path: str, dataset_type: str, is_train: bool, method: str):
        """Load train dataset"""
        trans = self.transforms_train if is_train else self.transforms_eval
        dataset_class = get_custom_dataset_class(dataset_type=dataset_type, method=method)
        if is_train:
            self.train_dataset = dataset_class(root=split_path, transform=trans)
        else:
            self.val_dataset = dataset_class(root=split_path, transform=trans)
