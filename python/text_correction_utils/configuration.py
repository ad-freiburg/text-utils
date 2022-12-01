import os
import re
from typing import Any

import yaml


def _replace_files(s: Any, base_dir: str) -> Any:
    file_ref_regex = re.compile(r"file\((.+\.yaml)\)")

    if isinstance(s, list):
        new_s = []
        for v in s:
            new_s.append(_replace_files(v, base_dir))
        return new_s
    elif isinstance(s, dict):
        new_dict = {}
        for k, v in s.items():
            new_dict[k] = _replace_files(v, base_dir)
        return new_dict
    elif isinstance(s, str):
        match = file_ref_regex.fullmatch(s)
        if match is not None:
            s = load_config(os.path.join(base_dir, match.group(1)))
        return s
    else:
        return s


def _replace_env_vars(s: str) -> str:
    orig_len = len(s)
    env_var_regex = re.compile(r"env\(([A-Z0-9_]+):?(.*?)\)")

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


def _replace_abs_paths(s: str) -> str:
    orig_len = len(s)
    path_regex = re.compile(r"abspath\((.+)\)")

    length_change = 0
    for match in re.finditer(path_regex, s):
        path = match.group(1)
        path = os.path.abspath(path)
        lower_idx = match.start() + length_change
        upper_idx = match.end() + length_change
        s = s[:lower_idx] + path + s[upper_idx:]
        length_change = len(s) - orig_len
    return s


def load_config(yaml_path: str) -> Any:
    """

    Loads a yaml config.
    Supports the following special operators:
        - env(ENV_VAR:default) for using environment variables with optional default values
        - file(relative/path/file.yaml) for loading other yaml files relative to current file
        - abspath(some/path) for turning paths into absolute paths

    :param yaml_path: path to config file
    :return: fully resolved yaml configuration

    >>> import os
    >>> os.environ["TEST_ENV_VAR"] = "123"
    >>> load_config("resources/test/test_config.yaml") # doctest: +NORMALIZE_WHITESPACE
    {'test': [123, '123', 123, 123],
    'subconfig': ['item1', 'item2', 'item3', {'test': 123}]}
    """
    with open(yaml_path, "r", encoding="utf8") as inf:
        raw_yaml = inf.read()

    raw_yaml = _replace_env_vars(raw_yaml)
    raw_yaml = _replace_abs_paths(raw_yaml)
    parsed_yaml = yaml.load(raw_yaml, Loader=yaml.FullLoader)
    base_dir = os.path.abspath(os.path.dirname(yaml_path))
    parsed_yaml = _replace_files(parsed_yaml, base_dir)
    return parsed_yaml
