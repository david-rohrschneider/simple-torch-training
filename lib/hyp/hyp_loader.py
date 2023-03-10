import os
from configparser import ConfigParser
from pathlib import Path

from lib.helpers.allowed_config_values import TRUE_VALUES, NONE_VALUES, OPTIMIZERS, \
    AUTO_AUG_POLICIES, INTERPOLATION_MODES, SCHEDULER_TYPES


class HypLoader:
    """Loader for all hyperparameters used for training"""

    __PROJECT_ROOT = Path(__file__).parent.parent.parent
    __DEFAULT_HYP_CONFIG = os.path.join(str(__PROJECT_ROOT), "configs/DEFAULT_HYP.ini")

    def __init__(self, hyp_config_path: str):
        """Constructor for HypLoader"""

        self.__hyp_config = ConfigParser()
        self.__hyp_config.read(self.__DEFAULT_HYP_CONFIG)
        if hyp_config_path is not None:
            self.__hyp_config.read(hyp_config_path)

        # Basic
        self.optimizer = None
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.norm_weight_decay = None
        self.clip_grad_norm = None
        self.bias_weight_decay = None
        self.label_smoothing = None

        # Transformers
        self.embedding_decay = None

        # Augmentation
        self.mixup_alpha = None
        self.cutmix_alpha = None
        self.auto_augment = None
        self.magnitude = None
        self.aug_mix_severity = None
        self.random_erase_prob = None
        self.interpolation_mode = None
        self.train_crop_size = None
        self.eval_crop_size = None
        self.eval_resize = None
        self.repeated_aug = None
        self.repeated_aug_reps = None

        # LR Scheduler
        self.scheduler_type = None
        self.warmup_epochs = None
        self.warmup_method = None
        self.warmup_decay = None
        self.scheduler_step_size = None
        self.scheduler_gamme = None
        self.min_lr = None

        # Distributed
        self.world_size = None
        self.warmup_method = None
        self.ema = None
        self.ema_steps = None
        self.ema_decay = None

        self.__init_values()

    def __get_config_value(self, section: str, key: str, number_type: type = float):
        config_value = self.__hyp_config[section][key]
        if config_value in NONE_VALUES:
            return None
        try:
            value = number_type(config_value)
        except ValueError as exc:
            raise ValueError(f"{key} must be of type {number_type} or None") from exc
        if value < 0:
            raise ValueError(f"{key} must be >= 0!")
        return value

    def __verify_optimizer(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"Optimizer has to be one of: {', '.join(OPTIMIZERS)}")

    def __verify_auto_augment(self):
        if self.auto_augment in NONE_VALUES:
            self.auto_augment = None
            return
        if self.auto_augment not in AUTO_AUG_POLICIES:
            raise ValueError(f"Auto augment policy must be one of: {', '.join(AUTO_AUG_POLICIES)}")

    def __verify_interpolation_mode(self):
        if self.interpolation_mode not in INTERPOLATION_MODES:
            raise ValueError(f"Interpolation mode must be one of: {', '.join(INTERPOLATION_MODES)}")

    def __verify_scheduler_type(self):
        if self.scheduler_type not in SCHEDULER_TYPES:
            raise ValueError(f"Scheduler type must be one of: {', '.join(SCHEDULER_TYPES)}")

    def __init_values(self):
        # Basic
        self.optimizer: str = self.__hyp_config["BASIC"]["Optimizer"].lower()
        self.__verify_optimizer()
        self.learning_rate: float = self.__get_config_value("BASIC", "LearningRate")
        self.momentum: float = self.__get_config_value("BASIC", "Momentum")
        self.weight_decay: float = self.__get_config_value("BASIC", "WeightDecay")
        self.norm_weight_decay: float = self.__get_config_value("BASIC", "NormWeightDecay")
        self.clip_grad_norm: float = self.__get_config_value("BASIC", "ClipGradNorm")
        self.bias_weight_decay: float = self.__get_config_value("BASIC", "BiasWeightDecay")
        self.label_smoothing: float = self.__get_config_value("BASIC", "LabelSmoothing")

        # Transformers
        self.embedding_decay: float = self.__get_config_value("TRANSFORMERS", "EmbeddingDecay")

        # Augmentation
        self.mixup_alpha: float = self.__get_config_value("AUGMENTATION", "MixupAlpha")
        self.cutmix_alpha: float = self.__get_config_value("AUGMENTATION", "CutmixALpha")
        self.auto_augment: str = self.__hyp_config["AUGMENTATION"]["AutoAugment"].lower()
        self.__verify_auto_augment()
        self.magnitude: int = self.__get_config_value("AUGMENTATION", "Magnitude", int)
        self.aug_mix_severity: int = self.__get_config_value("AUGMENTATION", "AugmixSeverity",
                                                             int)
        self.random_erase_prob: float = self.__get_config_value("AUGMENTATION", "RandomErase")
        self.interpolation_mode: str = \
            self.__hyp_config["AUGMENTATION"]["InterpolationMode"].lower()
        self.__verify_interpolation_mode()
        self.train_crop_size: int = self.__get_config_value("AUGMENTATION", "TrainCropSize",
                                                            int)
        self.eval_crop_size: int = self.__get_config_value("AUGMENTATION", "EvalCropSize", int)
        self.eval_resize: int = self.__get_config_value("AUGMENTATION", "EvalResize", int)
        self.repeated_aug: bool = \
            self.__hyp_config["AUGMENTATION"]["RepeatedAugmentation"] in TRUE_VALUES
        self.repeated_aug_reps: int = \
            self.__get_config_value("AUGMENTATION", "RepeatedAugmentationReps", int)

        # LR Scheduler
        self.scheduler_type: str = self.__hyp_config["LR_SCHEDULER"]["Type"].lower()
        self.__verify_scheduler_type()
        self.warmup_epochs: int = self.__get_config_value("LR_SCHEDULER", "WarmupEpochs", int)
        self.warmup_method: str = self.__hyp_config["LR_SCHEDULER"]["WarmupMethod"].lower()
        self.warmup_decay: float = self.__get_config_value("LR_SCHEDULER", "WarmupDecay")
        self.scheduler_step_size: int = self.__get_config_value("LR_SCHEDULER", "StepSize", int)
        self.scheduler_gamme: float = self.__get_config_value("LR_SCHEDULER", "Gamma")
        self.min_lr: float = self.__get_config_value("LR_SCHEDULER", "Min")

        # Distributed
        self.world_size: int = self.__get_config_value("DISTRIBUTED", "WorldSize", int)
        self.warmup_method: str = self.__hyp_config["DISTRIBUTED"]["Url"]
        self.ema: bool = self.__hyp_config["DISTRIBUTED"]["EMA"] in TRUE_VALUES
        self.ema_steps: int = self.__get_config_value("DISTRIBUTED", "EmaSteps", int)
        self.ema_decay: float = self.__get_config_value("DISTRIBUTED", "EmaDecay")
