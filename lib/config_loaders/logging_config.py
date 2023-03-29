from lib.config_loaders.config_loader import ConfigLoader
from lib.helpers import constants


class LoggingConfig(ConfigLoader):
    """Loader for logging config used for training"""

    __CONFIG_FILE = "LOGGING.ini"

    def __init__(self, custom: bool):
        """Constructor for HypLoader"""

        super().__init__(self.__CONFIG_FILE, custom)

        self.tb_logging: bool = self.get_bool("GENERAL", "Tensorboard")
        self.write_to_file: bool = self.get_bool("GENERAL", "WriteToFile")
        self.log_args: bool = self.get_bool("GENERAL", "LogArgs")
        self.log_model_overview: bool = self.get_bool("GENERAL", "LogModelOverview")
        self.log_hyperparams: bool = self.get_bool("GENERAL", "LogHyperparams")
        self.log_data: bool = self.get_bool("GENERAL", "LogData")

        # Training
        self.print_freq_train: int = self.get_int("TRAINING", "PrintFrequency")
        self.tb_log_type_train: str = self.get_str("TRAINING", "TensorboardLogType")

        # Evaluation
        self.print_freq_eval: int = self.get_int("EVALUATION", "PrintFrequency")
        self.tb_log_type_eval: str = self.get_str("EVALUATION", "TensorboardLogType")

        # Checkpoint
        self.ckpt_save_type: str = self.get_str("CHECKPOINT", "SaveType")
        self.last_n_ckpts: int = self.get_int("CHECKPOINT", "KeepLastNCheckpoints")

        self.__verify()

    def __verify_print_freq_train(self):
        if self.print_freq_train and self.print_freq_train <= 0:
            raise ValueError("Training print frequency cannot be <= 0!")

    def __verify_print_freq_eval(self):
        if self.print_freq_eval and self.print_freq_eval <= 0:
            raise ValueError("Evaluation print frequency cannot be <= 0!")

    def __verify_tb_log_type_train(self):
        if self.tb_log_type_train not in constants.TB_LOG_TYPES:
            raise ValueError("Tensorboard Logging type for training can only be "
                             f"{' or '.join(constants.TB_LOG_TYPES)}")

    def __verify_tb_log_type_eval(self):
        if self.tb_log_type_eval not in constants.TB_LOG_TYPES:
            raise ValueError("Tensorboard Logging type for evaluation can only be "
                             f"{' or '.join(constants.TB_LOG_TYPES)}")

    def __verify_ckpt_save_type(self):
        if self.ckpt_save_type not in constants.CKPT_SAVE_TYPES:
            raise ValueError("Checkpoint save type must be one of "
                             f"{', '.join(constants.CKPT_SAVE_TYPES)}")

    def __verify_last_n_ckpts(self):
        if self.ckpt_save_type == "LastN":
            if self.last_n_ckpts <= 0:
                raise ValueError("Last N Checkpoints must be > 0!")

    def __verify(self):
        self.__verify_print_freq_train()
        self.__verify_print_freq_eval()
        self.__verify_tb_log_type_train()
        self.__verify_ckpt_save_type()
        self.__verify_last_n_ckpts()
