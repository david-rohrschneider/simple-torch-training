from lib.helpers import constants
from lib.config_loaders.config_loader import ConfigLoader


class DataConfig(ConfigLoader):
    """Loader for all data loading settings used for training"""

    __CONFIG_FILE = "DATA.ini"

    def __init__(self, custom: bool):
        """Constructor for DataConfig"""

        super().__init__(self.__CONFIG_FILE, custom)

        # Normalization
        self.normalize: bool = self.get_bool("NORMALIZATION", "Normalize")
        self.norm_mean: tuple = self.get_tuple("NORMALIZATION", "Mean")
        self.norm_std: tuple = self.get_tuple("NORMALIZATION", "Std")

        # Mix up
        self.mix_up: bool = self.get_bool("MIX_UP", "Mixup")
        self.mix_up_alpha: float = self.get_float("MIX_UP", "MixupAlpha")
        self.cut_mix_alpha: float = self.get_float("MIX_UP", "CutmixALpha")

        # Auto augmentation
        self.auto_augment: bool = self.get_bool("AUTO_AUGMENT", "AutoAugment")
        self.auto_augment_policy: str = self.get_str("AUTO_AUGMENT", "AutoAugmentPolicy")
        self.magnitude: int = self.get_int("AUTO_AUGMENT", "Magnitude")
        self.severity: int = self.get_int("AUTO_AUGMENT", "Severity")

        # Flipping
        self.hor_flipping: bool = self.get_bool("FLIP", "HorizontalFlipping")
        self.h_flip_prob: bool = self.get_float("FLIP", "HFlipProb")

        # Cropping
        self.interpolation_mode: str = self.get_str("CROP", "InterpolationMode", "bilinear")
        self.crop: bool = self.get_bool("CROP", "Crop")
        self.train_crop_size: tuple = self.get_tuple("CROP", "TrainCropSize")
        self.eval_crop_size: tuple = self.get_tuple("CROP", "EvalCropSize")
        self.resize: bool = self.get_bool("CROP", "Resize")
        self.eval_resize: tuple = self.get_tuple("CROP", "EvalResize")
        self.repeated_aug: bool = self.get_bool("CROP", "RepeatedAugmentation")
        self.repeated_aug_reps: int = self.get_int("CROP", "RepeatedAugmentationReps")

        # Other
        self.random_erase_prob: float = self.get_float("AUGMENTATION", "RandomErase")

        self.__verify()

    def __verify_auto_augment(self):
        if self.auto_augment:
            return
        if self.auto_augment_policy and self.auto_augment_policy not in constants.AUTO_AUG_POLICIES:
            raise ValueError("Auto augment policy must be one of: "
                             f"{', '.join(constants.AUTO_AUG_POLICIES)}")

    def __verify_interpolation_mode(self):
        if self.interpolation_mode not in constants.INTERPOLATION_MODES:
            raise ValueError("Interpolation mode must be one of: "
                             f"{', '.join(constants.INTERPOLATION_MODES)}")

    def __verify(self):
        self.__verify_auto_augment()
        self.__verify_interpolation_mode()
