import os.path
import sys
from pathlib import Path
from datetime import datetime
from configparser import ConfigParser

from lib.args.args_loader import ArgsLoader
from lib.helpers.allowed_config_values import TB_LOG_TYPES, CKPT_SAVE_TYPES
from lib.helpers.dir_names import LogDirNames, LogFileNames
from lib.helpers.log_messages import \
    Processes, Prefixes, welcome_message, copyright_message, log_file_intro
from lib.hyp.hyp_loader import HypLoader


class TrainLogger:
    """Logger for training"""

    __PROJECT_ROOT = Path(__file__).parent.parent.parent
    __LOGGING_CONFIG = str(__PROJECT_ROOT) + "/configs/LOGGING.ini"

    def __init__(self):
        """Constructor for TrainLogger"""

        logging_config = ConfigParser()
        logging_config.read(self.__LOGGING_CONFIG)

        self.__print_welcome_message()
        self.__log_file_created: bool = False

        # General
        self.__tb_logging: bool = logging_config["GENERAL"]["Tensorboard"] == 'True'
        self.__write_to_file: bool = logging_config["GENERAL"]["WriteToFile"] == 'True'
        self.__log_args: bool = logging_config["GENERAL"]["LogArgs"] == 'True'
        self.__log_model_overview: bool = logging_config["GENERAL"]["LogModelOverview"] == 'True'
        self.__log_hyperparams: bool = logging_config["GENERAL"]["LogHyperparams"] == 'True'

        # Training
        self.__print_freq_train: int = int(logging_config["TRAINING"]["PrintFrequency"])
        self.__tb_log_type_train: str = logging_config["TRAINING"]["TensorboardLogType"].lower()

        # Evaluation
        self.__print_freq_eval: int = int(logging_config["EVALUATION"]["PrintFrequency"])
        self.__tb_log_type_eval: str = logging_config["EVALUATION"]["TensorboardLogType"].lower()

        # Checkpoint
        self.__ckpt_save_type: str = logging_config["CHECKPOINT"]["SaveType"].lower()
        self.__last_n_ckpts: int = int(logging_config["CHECKPOINT"]["KeepLastNCheckpoints"])

        self.__verify()

        self.__output_root_dir: str = ""
        self.__train_root_dir: str = ""

    def __verify_print_freq_train(self):
        if self.__print_freq_train <= 0:
            raise ValueError("Training print frequency cannot be <= 0!")

    def __verify_print_freq_eval(self):
        if self.__print_freq_eval <= 0:
            raise ValueError("Evaluation print frequency cannot be <= 0!")

    def __verify_tb_log_type(self):
        if self.__tb_log_type_train not in TB_LOG_TYPES or \
                self.__tb_log_type_train not in TB_LOG_TYPES:
            raise ValueError("Tensorboard Logging type can only be "
                             f"{'or'.join(TB_LOG_TYPES)}")

    def __verify_ckpt_save_type(self):
        if self.__ckpt_save_type not in CKPT_SAVE_TYPES:
            raise ValueError(f"Checkpoint save type must be one of {', '.join(CKPT_SAVE_TYPES)}")

    def __verify_last_n_ckpts(self):
        if self.__ckpt_save_type == "LastN":
            if self.__last_n_ckpts <= 0:
                raise ValueError("Last N Checkpoints must be > 0!")

    def __verify(self):
        try:
            self.__verify_print_freq_train()
            self.__verify_print_freq_eval()
            self.__verify_tb_log_type()
            self.__verify_ckpt_save_type()
            self.__verify_last_n_ckpts()
        except ValueError as exc:
            self.log_error(
                process=Processes.CONF_PARSING_LOG,
                exception_name=type(exc).__name__,
                message=exc.args[0])

    def __log(self, message: str):
        print(message)
        if self.__write_to_file and self.__log_file_created:
            with open(self.__log_file, mode="a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
            log_file.close()

    def log_error(self, process: Processes, exception_name: str, message: str,
                  exit_process: bool = True):
        self.__log(
            f"{Prefixes.ERROR.value}    Something went wrong during {process.value}!")
        self.__log(
            f"{Prefixes.ERROR.value}    ➡  {exception_name}: {message}")
        if exit_process:
            sys.exit(1)

    def log_warning(self, message: str):
        self.__log(f"{Prefixes.WARNING.value}    {message}")

    def log_info(self, message: str, show_date_time: bool = True):
        info_str = Prefixes.INFO.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def log_success(self, message: str, show_date_time: bool = True):
        info_str = Prefixes.SUCCESS.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def log_saving(self, message: str, show_date_time: bool = True):
        info_str = Prefixes.SAVING.value
        if show_date_time:
            info_str += f"[{datetime.now()}]    "
        else:
            info_str += "    "

        info_str += message
        self.__log(info_str)

    def __log_config(self, message: str):
        self.__log(f"{Prefixes.CONFIG.value}    {message}")

    def log_files(self, message: str, show_date_time: bool = True):
        info_str = Prefixes.FILES.value
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
                self.__log(f"{Prefixes.CONFIG.value}        ▫ {attr}: {value}")
        self.__log("\n")

    def init_logging(self, args_loader: ArgsLoader, hyp_loader: HypLoader):
        self.__init_output_dir(args_loader.output_dir)
        self.__init_new_train_dir()
        self.__init_log_file()

        if self.__log_args:
            self.__log_config("Loaded arguments:")
            self.__log_attributes(args_loader)
        if self.__log_hyperparams:
            self.__log_config("Loaded hyperparameter config:")
            self.__log_attributes(hyp_loader)

    def __init_output_dir(self, out_dir: str):
        if not os.path.exists(out_dir):
            self.log_warning("Output directory does not exist!")
            os.mkdir(out_dir)
            self.log_files(f"Created directory: {out_dir}")

        self.__output_root_dir = os.path.join(out_dir, LogDirNames.OUTPUT_ROOT.value)
        if not os.path.exists(self.__output_root_dir):
            os.mkdir(self.__output_root_dir)
            self.log_files(f"Created directory: {self.__output_root_dir}")

    def __init_new_train_dir(self):
        train_dir = os.path.join(self.__output_root_dir, LogDirNames.TRAINING_ROOT.value)
        train_try = 1
        while os.path.exists(f"{train_dir}{train_try}"):
            train_try += 1
        self.__train_root_dir = f"{train_dir}{train_try}"
        os.mkdir(self.__train_root_dir)
        self.log_files(f"Outputs will be saved to: {self.__train_root_dir}")

    def __init_log_file(self):
        self.__log_file = os.path.join(self.__train_root_dir, LogFileNames.TRAIN_LOG.value)
        if not os.path.isfile(self.__log_file):
            with open(self.__log_file, "w", encoding="utf-8") as file:
                file.write(f"{log_file_intro} - {datetime.now().strftime('%b %d %Y %H:%M:%S')}\n\n")
        self.log_files(f"Training log will be saved to: {self.__log_file}")
        self.__log_file_created = True

    @staticmethod
    def __print_welcome_message():
        width = len(welcome_message) + 15
        print("-" * width)
        print(" " * 6 + welcome_message)
        print(" " * 14 + copyright_message)
        print("-" * width)
