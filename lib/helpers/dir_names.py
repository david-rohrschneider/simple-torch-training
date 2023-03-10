from enum import Enum


class LogDirNames(Enum):
    OUTPUT_ROOT = "Output"
    TRAINING_ROOT = "Train_"
    CHECKPOINTS = "Checkpoints"
    EVAL = "Evaluation"
    EVAL_ROOT = "Eval_"


class LogFileNames(Enum):
    TRAIN_LOG = "TRAIN_LOG.txt"


class DataPaths(Enum):
    TRAIN = "train"
    VALID = "val"
