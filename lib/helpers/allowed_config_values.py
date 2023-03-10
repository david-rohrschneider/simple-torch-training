from torchvision.transforms import AutoAugmentPolicy
from torchvision import models

# General
NONE_VALUES = ["None", "none", "NONE"]
TRUE_VALUES = ["True", "true", "TRUE"]

# Hyp config
OPTIMIZERS = ["adadelta", "adagrad", "adamw", "sparseadam", "adamax", "asgd",
              "lbfgs", "nadam", "radam", "rmsprop", "rprop", "sgd"]
SCHEDULER_TYPES = ["lambdalr", "multiplicativelr", "steplr", "multisteplr", "constantlr",
                   "linearlr", "exponentiallr", "polynomiallr", "cosineannealinglr",
                   "chainedscheduler", "sequentiallr", "reducelronplateau", "cycliclr",
                   "onecyclelr", "cosineannealingwarmrestarts"]
WARMUP_METHODS = ["linear", "constant"]
AUTO_AUG_POLICIES = [p.value for p in AutoAugmentPolicy]
INTERPOLATION_MODES = ["nearest", "linear", "bilinear", "bicubic", "trilinear",
                       "area", "nearest-exact"]

# Args
METHODS = ["classification", "detection", "segmentation"]
TORCH_MODELS = models.list_models()

# Logging config
TB_LOG_TYPES = ["batch", "epoch"]
CKPT_SAVE_TYPES = ["all", "best", "lastn", "none"]
