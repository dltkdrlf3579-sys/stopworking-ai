from __future__ import annotations

import configparser
from pathlib import Path


class LenientConfigParser(configparser.ConfigParser):
    """ConfigParser that treats blank values as missing for typed getters."""

    def getint(
        self,
        section: str,
        option: str,
        *,
        raw: bool = False,
        vars=None,
        fallback=configparser._UNSET,
        **kwargs,
    ) -> int:
        value = self.get(section, option, raw=raw, vars=vars, fallback=fallback)
        if isinstance(value, str) and value.strip() == "" and fallback is not configparser._UNSET:
            return fallback
        return int(value)

    def getfloat(
        self,
        section: str,
        option: str,
        *,
        raw: bool = False,
        vars=None,
        fallback=configparser._UNSET,
        **kwargs,
    ) -> float:
        value = self.get(section, option, raw=raw, vars=vars, fallback=fallback)
        if isinstance(value, str) and value.strip() == "" and fallback is not configparser._UNSET:
            return fallback
        return float(value)

    def getboolean(
        self,
        section: str,
        option: str,
        *,
        raw: bool = False,
        vars=None,
        fallback=configparser._UNSET,
        **kwargs,
    ) -> bool:
        value = self.get(section, option, raw=raw, vars=vars, fallback=fallback)
        if isinstance(value, str) and value.strip() == "" and fallback is not configparser._UNSET:
            return fallback
        if isinstance(value, bool):
            return value
        return super()._convert_to_boolean(str(value))


def load_config(path: str = "config.ini") -> configparser.ConfigParser:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    config = LenientConfigParser()
    config.read(config_path, encoding="utf-8")
    return config


def getint(config: configparser.ConfigParser, section: str, option: str, default: int) -> int:
    return config.getint(section, option, fallback=default)


def getfloat(config: configparser.ConfigParser, section: str, option: str, default: float) -> float:
    return config.getfloat(section, option, fallback=default)


def getbool(config: configparser.ConfigParser, section: str, option: str, default: bool) -> bool:
    return config.getboolean(section, option, fallback=default)
