import os

import torch
from torchvision import models

from lib.helpers import constants


class ArgsLoader:
    """Service for loading training arguments"""

    def __init__(self, args):
        """Constructor for ArgsLoader"""

        # General
        self.cuda: bool = args.cuda
        self.epochs: int = args.epochs
        self.start_epoch: int = args.start_epoch
        self.method: str = args.method.lower()
        self.amp: bool = args.amp
        self.eval: bool = args.eval
        self.sync_bn: bool = args.sync_bn
        self.output_dir: str = os.path.abspath(args.output_dir)
        self.distributed: bool = args.distributed

        # Configs
        self.custom_hyp_config: bool = args.custom_hyp_cfg
        self.custom_data_config: bool = args.custom_data_cfg

        # Data
        self.dataset: str = args.dataset.lower() if args.dataset else None
        self.dataset_type: str = args.dataset_type.lower()
        self.data_path: str = os.path.abspath(args.data_path)
        self.data_train: str = args.data_train
        self.data_val: str = args.data_val
        self.batch_size: int = args.batch_size
        self.workers: int = args.workers
        self.cache_dataset: bool = args.cache_dataset

        # Model
        self.model: str = args.model.lower()
        self.torch_hub_repo: str = args.torch_hub_repo
        self.torch_hub_pretrained: bool = args.torch_hub_pretrained
        self.weights_enum: str = args.weights_enum
        self.resume_from: str = args.resume_from

        # Verify that all args are correct
        self.__verify()

    def __verify_cuda(self):
        if self.cuda and not torch.cuda.is_available():
            raise ValueError("CUDA ist not available!")

    def __verify_epochs(self):
        if self.epochs <= 0:
            raise ValueError("Epochs cannot be <= 0!")

    def __verify_start_epoch(self):
        if self.start_epoch >= self.epochs:
            raise ValueError("Start epoch cannot be >= epochs!")

    def __verify_method(self):
        if self.method not in constants.METHODS:
            raise ValueError(f"Method can only be {'or'.join(constants.METHODS)}")

    def __verify_dataset_type(self):
        if self.dataset_type not in constants.DATASET_TYPES[self.method]:
            raise ValueError(f"For {self.method}, supported dataset types are "
                             f"{', '.join(constants.DATASET_TYPES[self.method])}")

    def __verify_data_path(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError("The specified data path does not exist!")
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError("The specified data path is not a directory!")

    def __verify_data_train(self):
        train_path = os.path.join(self.data_path, self.data_train)
        if not os.path.exists(train_path):
            raise FileNotFoundError("The specified train data path does not exist!")
        if not os.path.isdir(train_path):
            raise NotADirectoryError("The specified train data path is not a directory!")
        if len(os.listdir(train_path)) == 0:
            raise FileNotFoundError("The specified train data path is empty!")

    def __verify_data_val(self):
        val_path = os.path.join(self.data_path, self.data_val)
        if not os.path.exists(val_path):
            raise FileNotFoundError("The specified val data path does not exist!")
        if not os.path.isdir(val_path):
            raise NotADirectoryError("The specified val data path is not a directory!")
        if len(os.listdir(val_path)) == 0:
            raise FileNotFoundError("The specified val data path is empty!")

    def __verify_batch_size(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size cannot be <= 0!")

    def __verify_workers(self):
        if self.workers <= 0:
            raise ValueError("Workers cannot be <= 0!")

    def __verify_model(self):
        if self.torch_hub_repo:
            if self.model not in torch.hub.list(self.torch_hub_repo):
                raise ValueError("Specified model name not in models list of torch hub repo!")
        if self.model not in constants.TORCH_MODELS:
            raise ValueError("Specified model name not in torchvision models list!")

    def __verify_weights_enum(self):
        if self.weights_enum:
            try:
                models.get_weight(self.weights_enum)
            except ValueError as exc:
                raise ValueError("Specified weights enum not in torch weights list!") from exc

    def __verify_resume_from(self):
        if self.resume_from is None:
            return
        if not self.resume_from[-3:] == ".pt":
            raise ValueError("The specified checkpoint file is not a .pt file!")
        if not os.path.exists(self.resume_from):
            raise FileNotFoundError("The specified checkpoint file does not exist!")

    def __verify(self):
        self.__verify_cuda()
        self.__verify_epochs()
        self.__verify_start_epoch()
        self.__verify_method()
        self.__verify_dataset_type()
        self.__verify_data_path()
        self.__verify_data_train()
        self.__verify_data_val()
        self.__verify_batch_size()
        self.__verify_workers()
        self.__verify_model()
        self.__verify_weights_enum()
        self.__verify_resume_from()
