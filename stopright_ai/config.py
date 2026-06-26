from __future__ import annotations

import configparser
from pathlib import Path


def load_config(path: str = "config.ini") -> configparser.ConfigParser:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config


def getint(config: configparser.ConfigParser, section: str, option: str, default: int) -> int:
    return config.getint(section, option, fallback=default)


def getfloat(config: configparser.ConfigParser, section: str, option: str, default: float) -> float:
    return config.getfloat(section, option, fallback=default)


def getbool(config: configparser.ConfigParser, section: str, option: str, default: bool) -> bool:
    return config.getboolean(section, option, fallback=default)

