import os
import configparser
from pathlib import Path

from lib.helpers import constants


class ConfigLoader:
    """Super class for config loaders"""

    __PROJECT_ROOT = Path(__file__).parent.parent.parent
    __CONFIG_ROOT = os.path.join(str(__PROJECT_ROOT), "configs")
    __DEFAULT_CONFIG_ROOT = os.path.join(__CONFIG_ROOT, "default")

    def __init__(self, config_file_name: str, custom: bool):
        default_config = os.path.join(self.__DEFAULT_CONFIG_ROOT, config_file_name)
        self.__config = configparser.ConfigParser()
        self.__config.read(default_config)
        if custom:
            custom_config = os.path.join(self.__CONFIG_ROOT, config_file_name)
            self.__config.read(custom_config)

    def __is_none(self, section: str, key: str):
        if section not in self.__config:
            return True
        if key not in self.__config[section]:
            return True
        if self.__config[section][key].lower() in constants.NONE_VALUES:
            return True
        return False

    def get_str(self, section: str, key: str, fallback: str = None):
        if self.__is_none(section, key):
            if fallback is not None:
                return fallback
            return None

        return self.__config.get(section, key).lower()

    def get_bool(self, section: str, key: str, fallback: bool = None):
        if self.__is_none(section, key):
            if fallback is not None:
                return fallback
            return None
        try:
            value = self.__config.getboolean(section, key)
        except ValueError as exc:
            raise TypeError(f"{key} must be of type bool or None") from exc

        return value

    def get_int(self, section: str, key: str, fallback: int = None):
        if self.__is_none(section, key):
            if fallback is not None:
                return fallback
            return None
        try:
            value = self.__config.getint(section, key)
        except ValueError as exc:
            raise TypeError(f"{key} must be of type int or None") from exc

        if value < 0:
            raise ValueError(f"{key} must be >= 0!")

        return value

    def get_float(self, section: str, key: str, fallback: float = None):
        if self.__is_none(section, key):
            if fallback is not None:
                return fallback
            return None
        try:
            value = self.__config.getfloat(section, key)
        except ValueError as exc:
            raise TypeError(f"{key} must be of type float or None") from exc

        if value < 0:
            raise ValueError(f"{key} must be >= 0!")

        return value

    def get_tuple(self, section: str, key: str, fallback: tuple = None):
        if self.__is_none(section, key):
            if fallback is not None:
                return fallback
            return None
        value = self.__config.get(section, key)
        try:
            float_tuple = tuple([float(s.strip()) for s in value.split(",")])
        except ValueError:
            raise ValueError("Only int or float values (comma separated) are allowed for "
                             f"{key} in section {section}")
        return float_tuple
