from torchvision.transforms import AutoAugmentPolicy
from torchvision import models

# General
NONE_VALUES = ["none", "undefined", "null"]

# Hyp config
OPTIMIZERS = ["adadelta", "adagrad", "adamw", "sparseadam", "adamax", "asgd",
              "lbfgs", "nadam", "radam", "rmsprop", "rprop", "sgd"]
SCHEDULER_TYPES = ["lambdalr", "multiplicativelr", "steplr", "multisteplr", "constantlr",
                   "linearlr", "exponentiallr", "polynomiallr", "cosineannealinglr",
                   "chainedscheduler", "sequentiallr", "reducelronplateau", "cycliclr",
                   "onecyclelr", "cosineannealingwarmrestarts"]
WARMUP_METHODS = ["linear", "constant"]
AUTO_AUG_POLICIES = ["randomaugment", "trivialaugmentwide", "augmix"] + \
                    [p.value for p in AutoAugmentPolicy]
INTERPOLATION_MODES = ["nearest", "linear", "bilinear", "bicubic", "trilinear",
                       "area", "nearest-exact"]

CV2_INTERPOLATION_MODES = {
    "nearest": 0,
    "linear": 1,
    "bicubic": 2,
    "area": 3
}

# Args
METHODS = ["classification", "detection", "segmentation"]
TORCH_MODELS = ["custom"] + models.list_models()
DATASET_TYPES = {
    "classification": ["imagefolder", "custom"],
    "detection": ["coco", "voc", "custom"],
    "segmentation": ["coco", "voc", "custom"]
}

# Logging config
TB_LOG_TYPES = ["batch", "epoch"]
CKPT_SAVE_TYPES = ["all", "best", "lastn", "none"]
