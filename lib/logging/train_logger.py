import os.path
import sys
from datetime import datetime

from lib.args.args_loader import ArgsLoader
from lib.logging import log_messages
from lib.helpers import enums
from lib.config_loaders.logging_config import LoggingConfig
from lib.config_loaders.hyp_config import HypConfig
from lib.config_loaders.data_config import DataConfig


class TrainLogger:
    """Logger for training"""

    def __init__(self):
        """Constructor for TrainLogger"""

        self.__print_welcome_message()
        self.__log_file_created: bool = False

        self.__logging_config = None

        self.__output_root_dir: str = ""
        self.__train_root_dir: str = ""

    def load_config(self, custom: bool):
        try:
            self.__logging_config = LoggingConfig(custom=custom)
        except ValueError as exc:
            self.log_error(
                process=log_messages.Processes.CONF_PARSING_LOG,
                exception_name=type(exc).__name__,
                message=exc.args[0])

    def __log(self, message: str):
        print(message)
        if self.__logging_config is None:
            return
        if self.__logging_config.write_to_file and self.__log_file_created:
            with open(self.__log_file, mode="a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
            log_file.close()

    def log_error(self, process: log_messages.Processes, exception_name: str, message: str,
                  exit_process: bool = True):
        self.__log(
            f"{log_messages.Prefixes.ERROR.value}    Something went wrong during {process.value}!")
        self.__log(
            f"{log_messages.Prefixes.ERROR.value}    ➡  {exception_name}: {message}")
        if exit_process:
            sys.exit(1)

    def log_warning(self, message: str):
        self.__log(f"{log_messages.Prefixes.WARNING.value}    {message}")

    def log_info(self, message: str, show_date_time: bool = False):
        info_str = log_messages.Prefixes.INFO.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def log_success(self, message: str, show_date_time: bool = False):
        info_str = log_messages.Prefixes.SUCCESS.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def log_saving(self, message: str, show_date_time: bool = False):
        info_str = log_messages.Prefixes.SAVING.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def __log_config(self, message: str):
        self.__log(f"{log_messages.Prefixes.CONFIG.value}    {message}")

    def log_files(self, message: str, show_date_time: bool = False):
        info_str = log_messages.Prefixes.FILES.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def __log_attributes(self, attr_loader):
        class_dict = attr_loader.__dict__
        for attr, value in class_dict.items():
            if "_" not in attr[:1]:
                self.__log(f"{log_messages.Prefixes.CONFIG.value}        ▫ {attr}: {value}")
        self.__log("\n")

    def init_logging(self, args_loader: ArgsLoader, hyp_config: HypConfig, data_config: DataConfig):
        self.__init_output_dir(args_loader.output_dir)
        self.__init_new_train_dir()
        self.__init_log_file()

        if self.__logging_config.log_args:
            self.__log_config("Loaded arguments:")
            self.__log_attributes(args_loader)
        if self.__logging_config.log_hyperparams:
            self.__log_config("Loaded hyperparameter config:")
            self.__log_attributes(hyp_config)
        if self.__logging_config.log_data:
            self.__log_config("Loaded data config:")
            self.__log_attributes(data_config)

    def __init_output_dir(self, out_dir: str):
        if not os.path.exists(out_dir):
            self.log_warning("Output directory does not exist!")
            os.mkdir(out_dir)
            self.log_files(f"Created directory: {out_dir}")

        self.__output_root_dir = os.path.join(out_dir, dir_names.LogDirNames.OUTPUT_ROOT.value)
        if not os.path.exists(self.__output_root_dir):
            os.mkdir(self.__output_root_dir)
            self.log_files(f"Created directory: {self.__output_root_dir}")

    def __init_new_train_dir(self):
        train_dir = os.path.join(self.__output_root_dir, dir_names.LogDirNames.TRAINING_ROOT.value)
        train_try = 1
        while os.path.exists(f"{train_dir}{train_try}"):
            train_try += 1
        self.__train_root_dir = f"{train_dir}{train_try}"
        os.mkdir(self.__train_root_dir)
        self.log_files(f"Outputs will be saved to: {self.__train_root_dir}")

    def __init_log_file(self):
        self.__log_file = os.path.join(self.__train_root_dir,
                                       dir_names.LogFileNames.TRAIN_LOG.value)
        if not os.path.isfile(self.__log_file):
            with open(self.__log_file, "w", encoding="utf-8") as file:
                file.write(f"{log_messages.log_file_intro} - "
                           f"{datetime.now().strftime('%b %d %Y %H:%M:%S')}\n\n")
        self.log_files(f"Training log will be saved to: {self.__log_file}")
        self.__log_file_created = True

    @staticmethod
    def __print_welcome_message():
        width = len(log_messages.welcome) + 15
        print("-" * width)
        print(" " * 6 + log_messages.welcome)
        print(" " * 14 + log_messages.copyright)
        print("-" * width)
