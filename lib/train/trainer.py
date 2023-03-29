import os
import torch

from lib.args.args_loader import ArgsLoader
from lib.config_loaders.hyp_config import HypConfig
from lib.logging.train_logger import TrainLogger
from lib.data.data_loader import DataLoader
from lib.logging import log_messages
from lib.helpers import enums


class Trainer:
    """Loads data and model, performs training and eval"""

    def __init__(self, args):
        """Constructor for Trainer"""

        self.__logger: TrainLogger = TrainLogger()
        self.__logger.load_config(custom=args.custom_log_cfg)
        self.__args_loader: ArgsLoader = self.__create_args_loader(args)
        self.__create_cache()
        self.__hyp_config: HypConfig = self.__create_hyp_config()

        self.__device = torch.device("cuda" if self.__args_loader.cuda else "cpu")
        self.__data_loader: DataLoader = self.__create_data_loader()

        self.__logger.init_logging(self.__args_loader, self.__hyp_config, self.__data_loader.config)

    def __create_args_loader(self, args):
        self.__logger.log_info("Loading and verifying args...")
        try:
            args_loader = ArgsLoader(args=args)
            self.__logger.log_success("Loaded and verified args!")
            return args_loader
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            self.__logger.log_error(
                process=log_messages.Processes.ARG_PARSING,
                exception_name=type(exc).__name__,
                message=exc.args[0])
            return None

    def __create_cache(self):
        if not self.__args_loader.cache_dataset:
            return
        cache_root = os.path.expanduser(os.path.join("~", dir_names.CacheDirNames.ROOT.value))
        if not os.path.exists(cache_root):
            os.mkdir(cache_root)
        if self.__args_loader.cache_dataset:
            os.mkdir(os.path.join(cache_root, dir_names.CacheDirNames.DATASETS.value))

    def __create_hyp_config(self):
        self.__logger.log_info("Loading and verifying hyperparameter config...")
        if self.__args_loader.custom_hyp_config:
            self.__logger.log_info(
                "Using custom hyperparameter config, overwriting default values.")
        else:
            self.__logger.log_warning(
                "Using values from default hyperparameter config!")
        try:
            hyp_config = HypConfig(self.__args_loader.custom_hyp_config)
            self.__logger.log_success("Loaded and verified hyperparameters!")
            return hyp_config
        except (ValueError, TypeError) as exc:
            self.__logger.log_error(
                process=log_messages.Processes.CONF_PARSING_HYP,
                exception_name=type(exc).__name__,
                message=exc.args[0])
            return None

    def __create_data_loader(self):
        self.__logger.log_info("Loading and verifying data config...")
        if self.__args_loader.custom_data_config:
            self.__logger.log_info(
                "Using custom data config, overwriting default values.")
        else:
            self.__logger.log_warning(
                "Using values from default data config!")
        try:
            data_loader = DataLoader(self.__args_loader.custom_data_config, self.__device)
            self.__logger.log_success("Loaded and verified data config!")
            return data_loader
        except (ValueError, TypeError) as exc:
            self.__logger.log_error(
                process=log_messages.Processes.CONF_PARSING_DATA,
                exception_name=type(exc).__name__,
                message=exc.args[0])
            return None

    def __init_dataset(self, split_path: str, is_train: bool):
        data_path = os.path.join(self.__args_loader.data_path, split_path)
        cache_path = self.__data_loader.get_cache_path(data_path)
        if self.__args_loader.cache_dataset and os.path.exists(cache_path):
            self.__logger.log_files(f"Loading cached dataset from {cache_path}...")
            time = self.__data_loader.load_cached_dataset(cache_path=cache_path)
            self.__logger.log_files(f"Loaded cached dataset in {time}s!")
            return

        self.__logger.log_files(f"Loading dataset from {data_path}...")
        time = self.__data_loader.load_dataset(
            train_path=data_path,
            dataset_type=self.__args_loader.dataset_type,
            is_train=is_train)
        self.__logger.log_success(f"Loaded dataset in {time}s!")
        if self.__args_loader.cache_dataset:
            self.__logger.log_saving("Caching dataset...")
            self.__data_loader.cache_dataset(
                dir_path=data_path,
                cache_path=cache_path)
            self.__logger.log_success("Cached dataset!")
