import os
import re
from typing import Any, Dict, Union, List

import yaml

YamlConfig = Union[List[Any], Dict[str, Any]]


def _replace_files(s: YamlConfig, base_dir: str) -> YamlConfig:
    file_ref_regex = re.compile(r"\s*FILE\((.+\.yaml)\)\s*")

    if isinstance(s, list):
        new_s = []
        for v in s:
            if isinstance(s, str):
                match = file_ref_regex.fullmatch(s)
                if match is not None:
                    v = load_config(os.path.join(base_dir, match.group(1)))
            new_s.append(v)
        return new_s
    elif isinstance(s, dict):
        new_dict = {}
        for k, v in s.items():
            if isinstance(v, str):
                match = file_ref_regex.fullmatch(v)
                if match is not None:
                    v = load_config(os.path.join(base_dir, match.group(1)))
            new_dict[k] = v
        return new_dict
    else:
        return s


def _replace_env_vars(s: str) -> str:
    orig_len = len(s)
    env_var_regex = re.compile(r"\s*ENV\(([A-Z_]+):?(.*?)\)\s*")

    length_change = 0
    for match in re.finditer(env_var_regex, s):
        env_var, env_default = match.groups()
        if env_var not in os.environ:
            if env_default == "":
                raise ValueError(f"environment variable {env_var} not found and no default was given")
            else:
                env_var = env_default
        else:
            env_var = os.environ[env_var]
        lower_idx = match.start() + length_change
        upper_idx = match.end() + length_change
        s = s[:lower_idx] + env_var + s[upper_idx:]
        length_change = len(s) - orig_len
    return s


def load_config(yaml_path: str) -> YamlConfig:
    with open(yaml_path, "r", encoding="utf8") as inf:
        raw_yaml = inf.read()

    raw_yaml = _replace_env_vars(raw_yaml)
    parsed_yaml = yaml.load(raw_yaml, Loader=yaml.FullLoader)
    parsed_yaml = _replace_files(parsed_yaml, os.path.abspath(os.path.dirname(yaml_path)))
    return parsed_yaml
