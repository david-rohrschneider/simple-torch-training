from lib.helpers import constants
from lib.config_loaders.config_loader import ConfigLoader


class HypConfig(ConfigLoader):
    """Loader for all hyperparameter settings used for training"""

    __CONFIG_FILE = "HYP.ini"

    def __init__(self, custom: bool):
        """Constructor for HypConfig"""

        super().__init__(self.__CONFIG_FILE, custom)

        # Basic
        self.optimizer: str = self.get_str("BASIC", "Optimizer")
        self.learning_rate: float = self.get_float("BASIC", "LearningRate")
        self.momentum: float = self.get_float("BASIC", "Momentum")
        self.weight_decay: float = self.get_float("BASIC", "WeightDecay")
        self.norm_weight_decay: float = self.get_float("BASIC", "NormWeightDecay")
        self.clip_grad_norm: float = self.get_float("BASIC", "ClipGradNorm")
        self.bias_weight_decay: float = self.get_float("BASIC", "BiasWeightDecay")
        self.label_smoothing: float = self.get_float("BASIC", "LabelSmoothing")

        # Transformers
        self.embedding_decay: float = self.get_float("TRANSFORMERS", "EmbeddingDecay")

        # LR Scheduler
        self.scheduler_type: str = self.get_str("LR_SCHEDULER", "Type")
        self.warmup_epochs: int = self.get_int("LR_SCHEDULER", "WarmupEpochs")
        self.warmup_method: str = self.get_str("LR_SCHEDULER", "WarmupMethod")
        self.warmup_decay: float = self.get_float("LR_SCHEDULER", "WarmupDecay")
        self.scheduler_step_size: int = self.get_int("LR_SCHEDULER", "StepSize")
        self.scheduler_gamma: float = self.get_float("LR_SCHEDULER", "Gamma")
        self.min_lr: float = self.get_float("LR_SCHEDULER", "Min")

        # Distributed
        self.world_size: int = self.get_int("DISTRIBUTED", "WorldSize")
        self.warmup_method: str = self.get_str("DISTRIBUTED", "Url")
        self.ema: bool = self.get_bool("DISTRIBUTED", "EMA")
        self.ema_steps: int = self.get_int("DISTRIBUTED", "EmaSteps")
        self.ema_decay: float = self.get_float("DISTRIBUTED", "EmaDecay")

        self.__verify()

    def __verify_optimizer(self):
        if self.optimizer not in constants.OPTIMIZERS:
            raise ValueError(f"Optimizer has to be one of: {', '.join(constants.OPTIMIZERS)}")

    def __verify_scheduler_type(self):
        if self.scheduler_type not in constants.SCHEDULER_TYPES:
            raise ValueError("Scheduler type must be one of: "
                             f"{', '.join(constants.SCHEDULER_TYPES)}")

    def __verify(self):
        self.__verify_optimizer()
        self.__verify_scheduler_type()
