import os
from contextlib import contextmanager
import logging
from threading import Lock
from typing import Dict, Any, Type, Union, Generator

import yaml
import torch
from flask import Flask, Response, cli, jsonify

from text_utils.api.processor import TextProcessor
from text_utils.api.utils import gpu_info, cpu_info
from text_utils.logging import get_logger
from text_utils import configuration


class Error:
    def __init__(self, msg: str, status: int):
        self.msg = msg
        self.status = status

    def to_response(self) -> Response:
        return Response(self.msg, status=self.status)


class TextProcessingServer:
    text_processor_cls: Type[TextProcessor]

    @classmethod
    def from_config(cls, path: str) -> "TextProcessingServer":
        config = configuration.load_config(path)
        return cls(config)

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            f"{self.text_processor_cls.task.upper()} SERVER"
        )
        self.logger.info(f"loaded server config:\n{yaml.dump(config)}")
        self.port = int(self.config.get("port", 40000))
        # disable flask startup message and set flask mode to development
        cli.show_server_banner = lambda *_: None
        os.environ["FLASK_DEBUG"] = "development"
        self.server = Flask(__name__)
        max_content_length = int(
            float(config.get("max_content_length", 1000.0)) * 1000.0
        )
        self.server.config["MAX_CONTENT_LENGTH"] = max_content_length
        self.max_models_per_gpu = max(1, config.get("max_models_per_gpu", 3))
        self.base_url = config.get("base_url", "")
        self.allow_origin = config.get("allow_origin", "*")
        self.timeout = float(config.get("timeout", 10.0))
        logging.getLogger("werkzeug").disabled = True
        self.num_gpus = torch.cuda.device_count()

        assert "models" in config and len(config["models"]) > 0, \
            "expected at least one model to be specified in the server config"

        @self.server.after_request
        def _after_request(response: Response) -> Response:
            response.headers.add(
                "Access-Control-Allow-Origin",
                self.allow_origin
            )
            response.headers.add(
                "Access-Control-Allow-Headers",
                "*"
            )
            response.headers.add(
                "Access-Control-Allow-Private-Network",
                "true"
            )
            return response

        @self.server.route(f"{self.base_url}/info")
        def _info() -> Response:
            response = jsonify({
                "gpu": [gpu_info(i) for i in range(self.num_gpus)],
                "cpu": cpu_info(),
                "timeout": self.timeout,
            })
            return response

        self.text_processors: Dict[str, TextProcessor] = {}
        self.name_to_text_processor: Dict[str, str] = {}
        self.lock = Lock()

        model_duplicates = {}
        model_infos = []
        for cfg in config["models"]:
            if "device" in cfg:
                device = cfg["device"]
            elif self.num_gpus > 0:
                device = f"cuda:{len(self.text_processors) % self.num_gpus}"
            else:
                device = "cpu"

            if "name" in cfg:
                model_name = cfg["name"]
                org_model_name = model_name
                self.logger.info(
                    f"loading pretrained model {model_name} for task "
                    f"{self.text_processor_cls.task} onto device {device}"
                )
                text_processor = self.text_processor_cls.from_pretrained(
                    model_name,
                    device
                )
                model_info = next(
                    filter(
                        lambda m: m.name == model_name,
                        self.text_processor_cls.available_models()
                    ),
                    None
                )
                if model_info is None:
                    raise RuntimeError(
                        f"model {model_name} not found in available models"
                    )
                model_description = model_info.description
                model_name = model_info.name
                model_tags = model_info.tags
                model_tags.append("src::pretrained")

            elif "path" in cfg:
                path = cfg["path"]
                org_model_name = path
                self.logger.info(
                    f"loading model for task {self.text_processor_cls.task} "
                    f"from experiment {path} onto device {device}"
                )
                text_processor = self.text_processor_cls.from_experiment(
                    path,
                    device
                )
                model_name = text_processor.name
                model_description = "loaded from custom experiment"
                model_tags = ["src::custom"]
            else:
                raise RuntimeError(
                    "expected either name or path in model config"
                )

            # handle the case when two models have the same name
            if model_name in self.text_processors:
                dup_num = model_duplicates.get(model_name, 0) + 1
                self.logger.warning(
                    f"found another model with name {model_name}, "
                    f"renaming current one to {model_name}_{dup_num}"
                )
                model_duplicates[model_name] = dup_num
                model_name = f"{model_name}_{dup_num}"

            model_infos.append((model_name, model_description, model_tags))

            self.text_processors[model_name] = text_processor
            self.name_to_text_processor[org_model_name] = model_name

        @self.server.route(f"{self.base_url}/models")
        def _models() -> Response:
            response = jsonify({
                "task": self.text_processor_cls.task,
                "models": [
                    {"name": name, "description": description, "tags": tags}
                    for name, description, tags in model_infos
                ]
            })
            return response

    @contextmanager
    def text_processor(self, model_name: str) -> Generator[Union[TextProcessor, Error], None, None]:
        if model_name not in self.text_processors:
            yield Error(f"model {model_name} does not exist", 404)
            return
        acquired = self.lock.acquire(timeout=self.timeout)
        if not acquired:
            yield Error(f"failed to reserve model within {self.timeout}s", 503)
            return
        try:
            yield self.text_processors[model_name]
        finally:
            self.lock.release()

    def run(self):
        self.server.run("0.0.0.0", self.port, debug=False, use_reloader=False)
