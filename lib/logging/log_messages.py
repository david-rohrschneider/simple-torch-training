from enum import Enum


class Processes(Enum):
    ARG_PARSING = "Argument parsing"
    CONF_PARSING_LOG = "Logging config parsing"
    CONF_PARSING_HYP = "Hyperparameter config parsing"
    CONF_PARSING_DATA = "Data config parsing"
    DATA_LOADING = "Dataset loading"


class Prefixes(Enum):
    INFO = "[💬][INFO]   "
    WARNING = "[⚠️][WARNING]"
    ERROR = "[⛔][ERROR]  "
    CONFIG = "[🔧][CONFIG] "
    FILES = "[📁][FILES]    "
    SAVING = "[💾][SAVING]   "
    SUCCESS = "[✔️][SUCCESS]"


welcome: str = "✨ Welcome to Simple-Torch-Training! ✨"
copyright: str = "© by David Rohrschneider"

log_file_intro: str = "Simple-Torch-Training LOG"
