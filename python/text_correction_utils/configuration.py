import os
import re
import tempfile
import zipfile
from typing import Any, Callable

import yaml


def _replace_env(s: str, _: str) -> Any:
    env_regex = re.compile(r"env\(([A-Z0-9_]+):?(.*?)\)")
    org_length = len(s)
    length_change = 0
    for match in env_regex.finditer(s):
        env_var, env_default = match.group(1), match.group(2)
        if env_var not in os.environ:
            if env_default == "":
                raise ValueError(f"environment variable {env_var} not found and no default was given")
            else:
                env_var = env_default
        else:
            env_var = os.environ[env_var]
        s = s[:match.start() + length_change] + env_var + s[match.end() + length_change:]
        length_change = len(s) - org_length
    return yaml.load(s, Loader=yaml.FullLoader)


def _replace_non_env(s: str, base_dir: str) -> Any:
    file_regex = re.compile(r"^file\((.+\.yaml)\)$")
    path_regex = re.compile(r"^abspath\((.+)\)$")
    eval_regex = re.compile(r"^eval\((.+)\)$")
    file_regex_match = file_regex.fullmatch(s)
    path_regex_match = path_regex.fullmatch(s)
    eval_regex_match = eval_regex.fullmatch(s)
    num_matches = (
        (file_regex_match is not None) +
        (path_regex_match is not None) +
        (eval_regex_match is not None)
    )
    assert num_matches <= 1, f"more than one config command matches '{s}'"
    if file_regex_match is not None:
        file_path = file_regex_match.group(1)
        file_path = str(_replace_non_env(file_path, base_dir))
        return load_config(os.path.join(base_dir, file_path))
    elif path_regex_match is not None:
        path = path_regex_match.group(1)
        path = str(_replace_non_env(path, base_dir))
        return os.path.abspath(path)
    elif eval_regex_match is not None:
        expression = eval_regex_match.group(1)
        org_length = len(expression)
        length_change = 0
        for match in eval_regex.finditer(expression):
            replacement = _replace_non_env(match.group(1), base_dir)
            expression = (
                expression[:match.start(1) + length_change]
                + str(replacement)
                + expression[match.end(1) + length_change:]
            )
            length_change = len(expression) - org_length
        expression = str(_replace_non_env(expression, base_dir))
        return eval(expression)
    else:
        return s


def _handle_cfg(s: Any, base_dir: str, handle_fn: Callable[[str, str], Any]) -> Any:
    if isinstance(s, list):
        new_s = []
        for v in s:
            new_s.append(_handle_cfg(v, base_dir, handle_fn))
        return new_s
    elif isinstance(s, dict):
        new_dict = {}
        for k, v in s.items():
            new_dict[k] = _handle_cfg(v, base_dir, handle_fn)
        return new_dict
    elif isinstance(s, str):
        return handle_fn(s, base_dir)
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
    Inside eval() only env() operators are supported.

    :param yaml_path: path to config file
    :return: fully resolved yaml configuration

    >>> import os
    >>> os.environ["TEST_ENV_VAR"] = "123"
    >>> load_config("resources/test/test_config.yaml") # doctest: +NORMALIZE_WHITESPACE
    {'eval': 500,
    'subconfig': ['item1', 'item2', 'item3', {'test': 123}],
    'test': [123, 123, 123, 123]}
    """
    with open(yaml_path, "r", encoding="utf8") as inf:
        raw_yaml = inf.read()

    base_dir = os.path.abspath(os.path.dirname(yaml_path))
    parsed_yaml = yaml.load(raw_yaml, Loader=yaml.FullLoader)
    parsed_yaml = _handle_cfg(parsed_yaml, base_dir, _replace_env)
    parsed_yaml = _handle_cfg(parsed_yaml, base_dir, _replace_non_env)
    return yaml.load(yaml.dump(parsed_yaml), Loader=yaml.FullLoader)


def load_config_from_experiment(dir: str) -> Any:
    info = load_config(os.path.join(dir, "info.yaml"))
    if not os.path.exists(os.path.join(dir, "configs.zip")):
        return load_config(os.path.join(dir, info["config_name"]))

    with zipfile.ZipFile(os.path.join(dir, "configs.zip"), "r", zipfile.ZIP_DEFLATED) as inz:
        with tempfile.TemporaryDirectory() as tmp_dir:
            inz.extractall(tmp_dir)
            return load_config(os.path.join(tmp_dir, info["config_name"]))
