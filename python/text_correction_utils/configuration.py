import os
import re
from typing import Any

import yaml


def _handle_str(s: str, base_dir: Any) -> Any:
    file_regex = re.compile(r"^file\((.+\.yaml)\)$")
    env_regex = re.compile(r"^env\(([A-Z0-9_]+):?(.*?)\)$")
    path_regex = re.compile(r"^abspath\((.+)\)$")
    eval_regex = re.compile(r"^eval\((.+)\)$")
    file_regex_match = file_regex.fullmatch(s)
    env_regex_match = env_regex.fullmatch(s)
    path_regex_match = path_regex.fullmatch(s)
    eval_regex_match = eval_regex.fullmatch(s)
    num_matches = (
            (file_regex_match is not None) + (env_regex_match is not None)
            + (path_regex_match is not None) + (eval_regex_match is not None)
    )
    assert num_matches <= 1, f"more than one config command matches '{s}'"
    if file_regex_match is not None:
        file_path = file_regex_match.group(1)
        file_path = _handle_str(file_path, base_dir)
        assert isinstance(file_path, str), "file() operator input should be a string"
        return load_config(os.path.join(base_dir, file_path))
    elif env_regex_match is not None:
        env_var, env_default = env_regex_match.group(1), env_regex_match.group(2)
        env_var = _handle_str(env_var, base_dir)
        assert isinstance(env_var, str), "env() operator input should be a string"
        env_default = _handle_str(env_default, base_dir)
        assert isinstance(env_default, str), "env() operator default should be a string"
        if env_var not in os.environ:
            if env_default == "":
                raise ValueError(f"environment variable {env_var} not found and no default was given")
            else:
                env_var = env_default
        else:
            env_var = os.environ[env_var]
        return yaml.load(env_var, Loader=yaml.FullLoader)
    elif path_regex_match is not None:
        path = path_regex_match.group(1)
        path = _handle_str(path, base_dir)
        assert isinstance(path, str), "abspath() operator input should be a string"
        return os.path.abspath(path)
    elif eval_regex_match is not None:
        expression = eval_regex_match.group(1)
        expression = _handle_str(expression, base_dir)
        assert isinstance(expression, str), "eval() operator input should be a string"
        return eval(expression)
    else:
        return s


def _handle_cfg(s: Any, base_dir: str) -> Any:
    if isinstance(s, list):
        new_s = []
        for v in s:
            new_s.append(_handle_cfg(v, base_dir))
        return new_s
    elif isinstance(s, dict):
        new_dict = {}
        for k, v in s.items():
            new_dict[k] = _handle_cfg(v, base_dir)
        return new_dict
    elif isinstance(s, str):
        return _handle_str(s, base_dir)
    else:
        return s


def load_config(yaml_path: str) -> Any:
    """

    Loads a yaml config.
    Supports the following special operators:
        - env(ENV_VAR:default) for using environment variables with optional default values
        - file(relative/path/file.yaml) for loading other yaml files relative to current file
        - abspath(some/path) for turning paths into absolute paths
        - eval(expression) for evaluating python expressions
    Note that these special operators can be nested.

    :param yaml_path: path to config file
    :return: fully resolved yaml configuration

    >>> import os
    >>> os.environ["TEST_ENV_VAR"] = "123"
    >>> load_config("resources/test/test_config.yaml") # doctest: +NORMALIZE_WHITESPACE
    {'eval': 500,
    'subconfig': ['item1', 'item2', 'item3', {'test': 123}],
    'test': [123, '123', 123, 123]}
    """
    with open(yaml_path, "r", encoding="utf8") as inf:
        raw_yaml = inf.read()

    parsed_yaml = yaml.load(raw_yaml, Loader=yaml.FullLoader)
    base_dir = os.path.abspath(os.path.dirname(yaml_path))
    parsed_yaml = _handle_cfg(parsed_yaml, base_dir)
    return yaml.load(yaml.dump(parsed_yaml), Loader=yaml.FullLoader)
