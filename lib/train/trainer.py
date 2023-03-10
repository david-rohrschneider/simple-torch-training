import os
import torch

from lib.args.args_loader import ArgsLoader
from lib.helpers.dir_names import DataPaths
from lib.helpers.log_messages import Processes
from lib.hyp.hyp_loader import HypLoader
from lib.train.train_logger import TrainLogger
from lib.data.data_loader import DataLoader


class Trainer:
    """Loads data and model, performs training and eval"""

    def __init__(self, args):
        """Constructor for Trainer"""

        self.__logger: TrainLogger = TrainLogger()
        self.__args_loader: ArgsLoader = self.__create_args_loader(args)
        self.__hyp_loader: HypLoader = self.__create_hyp_loader()

        self.__device = torch.device("cuda" if self.__args_loader.cuda else "cpu")
        self.__data_loader: DataLoader = DataLoader(self.__device)

        self.__logger.init_logging(self.__args_loader, self.__hyp_loader)

    def __create_args_loader(self, args):
        self.__logger.log_info("Loading and verifying args...", show_date_time=False)
        try:
            args_loader = ArgsLoader(args=args)
            self.__logger.log_success("Loaded and verified args!")
            return args_loader
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            self.__logger.log_error(
                process=Processes.ARG_PARSING,
                exception_name=type(exc).__name__,
                message=exc.args[0])
            return None

    def __create_hyp_loader(self):
        self.__logger.log_info("Loading and verifying hyperparameter config...",
                               show_date_time=False)
        if self.__args_loader.hyp_config is not None:
            self.__logger.log_info(
                "Custom hyperparameter config found, overwriting the default with values with "
                f"{self.__args_loader.hyp_config}!",
                show_date_time=False)
        else:
            self.__logger.log_warning(
                "No custom hyperparameter config found, using values from default hyp config!")
        try:
            hyp_loader = HypLoader(self.__args_loader.hyp_config)
            self.__logger.log_success("Loaded and verified hyperparameters!")
            return hyp_loader
        except ValueError as exc:
            self.__logger.log_error(
                process=Processes.CONF_PARSING_HYP,
                exception_name=type(exc).__name__,
                message=exc.args[0])
            return None

    def __load_datasets(self): # TODO: load cache data in separate function!!!
        data_path_train = os.path.join(self.__args_loader.data_path, DataPaths.TRAIN.value)
        self.__logger.log_files("Loading train dataset...", show_date_time=False)
        msg_train, cache_path_train = self.__data_loader.load_train_dataset(
            train_path=data_path_train,
            cached=self.__args_loader.cache_dataset,
            train_crop_size=self.__hyp_loader.train_crop_size,
            auto_augment_policy=self.__hyp_loader.auto_augment,
            random_erase_prob=self.__hyp_loader.random_erase_prob,
            ra_magnitude=self.__hyp_loader.magnitude,
            augmix_severity=self.__hyp_loader.aug_mix_severity,
            interpolation=self.__hyp_loader.interpolation_mode)
        self.__logger.log_success(msg_train, show_date_time=False)
        if self.__args_loader.cache_dataset:
            self.__logger.log_saving("Caching train dataset...", show_date_time=False)
            self.__data_loader.cache_dataset(
                dir_path=data_path_train,
                cache_path=cache_path_train)
            self.__logger.log_success("Cached train dataset!", show_date_time=False)

        data_path_val = os.path.join(self.__args_loader.data_path, DataPaths.VALID.value)
