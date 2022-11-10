import os
import re
from typing import Dict, Any, Set

import yaml


class BaseConfig:
    required_arguments: Set[str] = set()

    @classmethod
    def _check_required(cls, d: Dict[str, Any]) -> None:
        if (d is None or len(d) == 0) and len(cls.required_arguments) > 0:
            raise ValueError(f"Got empty dictionary or None but found the required arguments. "
                             f"Please specify all the required arguments {cls.required_arguments}.")
        for arg in cls.required_arguments:
            if arg not in d:
                raise ValueError(f"Could not find required argument {arg} in {d.keys()}. "
                                 f"Please specify all the required arguments {cls.required_arguments}.")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseConfig":
        raise NotImplementedError()

    @classmethod
    def _rep_env_variables(cls, s: str) -> str:
        orig_s = s
        env_var_regex = re.compile(r"\${([A-Z_]+):?(.*?)}")

        length_change = 0
        for match in re.finditer(env_var_regex, s):
            env_var, env_default = match.groups()
            if env_var not in os.environ:
                if env_default == "":
                    raise ValueError(f"Environment variable {env_var} not found and no default was given")
                else:
                    env_var = env_default
            else:
                env_var = os.environ[env_var]
            lower_idx = match.start() + length_change
            upper_idx = match.end() + length_change
            s = s[:lower_idx] + env_var + s[upper_idx:]
            length_change = len(s) - len(orig_s)
        return s

    @classmethod
    def from_yaml(cls, filepath: str) -> "BaseConfig":
        with open(filepath, "r", encoding="utf8") as f:
            raw_yaml = f.read()
        raw_yaml_with_env_vars = cls._rep_env_variables(raw_yaml)
        parsed_yaml = yaml.load(raw_yaml_with_env_vars, Loader=yaml.FullLoader)
        config = cls.from_dict(parsed_yaml)
        return config

    def get_parsed_vars(self) -> Dict[str, Any]:
        var = vars(self)
        parsed_var: Dict[str, Any] = {}
        for k, v in sorted(var.items()):
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], BaseConfig):
                parsed_var[k] = [v_i.get_parsed_vars() for v_i in v]
            elif isinstance(v, BaseConfig):
                parsed_var[k] = v.get_parsed_vars()
            else:
                parsed_var[k] = v
        return parsed_var

    def __repr__(self) -> str:
        parsed_var = self.get_parsed_vars()
        return yaml.dump(parsed_var, default_flow_style=False)
