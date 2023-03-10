import os

from torch import cuda
from torchvision import models

from lib.helpers.allowed_config_values import METHODS, TORCH_MODELS


class ArgsLoader:
    """Service for loading training arguments"""

    def __init__(self, args):
        """Constructor for ArgsLoader"""

        # General
        self.cuda: bool = args.cuda
        self.epochs: int = args.epochs
        self.start_epoch: int = args.start_epoch
        self.method: str = args.method.lower()
        self.hyp_config: str = args.hyp_config
        self.amp: bool = args.amp
        self.eval: bool = args.eval
        self.sync_bn: bool = args.sync_bn
        self.output_dir: str = os.path.abspath(args.output_dir)

        # Data
        self.data_path: str = os.path.abspath(args.data_path)
        self.batch_size: int = args.batch_size
        self.workers: int = args.workers
        self.cache_dataset: bool = args.cache_dataset

        # Model
        self.model: str = args.model.lower()
        self.weights_enum: str = args.weights_enum
        self.resume_from: str = args.resume_from

        # Verify that all args are correct
        self.__verify()

    def __verify_cuda(self):
        if self.cuda and not cuda.is_available():
            raise ValueError("CUDA ist not available!")

    def __verify_epochs(self):
        if self.epochs <= 0:
            raise ValueError("Epochs cannot be <= 0!")

    def __verify_start_epoch(self):
        if self.start_epoch >= self.epochs:
            raise ValueError("Start epochs cannot be >= epochs!")

    def __verify_method(self):
        if self.method not in METHODS:
            raise ValueError(f"Method can only be {'or'.join(METHODS)}")

    def __verify_hyp_config(self):
        if self.hyp_config is None:
            return
        if not self.hyp_config[-4:] == ".ini":
            raise ValueError("The specified config file is no .ini file!")
        if not os.path.exists(os.path.abspath(self.hyp_config)):
            raise FileNotFoundError("The specified config file does not exist!")

    def __verify_data_path(self):
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError("The specified data path is not a directory!")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError("The specified data path does not exist!")

    def __verify_batch_size(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size cannot be <= 0!")

    def __verify_workers(self):
        if self.workers <= 0:
            raise ValueError("Workers cannot be <= 0!")

    def __verify_model(self):
        if self.model not in TORCH_MODELS:
            raise ValueError("Specified model name not in torch models list!")

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
        self.__verify_hyp_config()
        self.__verify_data_path()
        self.__verify_batch_size()
        self.__verify_workers()
        self.__verify_model()
        self.__verify_weights_enum()
        self.__verify_resume_from()
