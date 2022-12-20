import os
import pprint
import logging
import json
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple, Set, Union

import torch
from torch.cuda import Stream
from flask import Flask, Response, abort, cli, jsonify, request

from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils.api.utils import gpu_info, cpu_info
from text_correction_utils.logging import get_logger


class TextCorrectionServer:
    text_corrector_classes: List[TextCorrector]

    @staticmethod
    def from_config(path: str) -> "TextCorrectionServer":
        with open(path, "r", encoding="utf8") as inf:
            config = json.loads(inf.read())
        return TextCorrectionServer(config)

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port = int(self.config.get("port", 12345))
        # disable flask startup message and set flask mode to development
        cli.show_server_banner = lambda *_: None
        os.environ["FLASK_ENV"] = "development"
        self.server = Flask(__name__)
        max_content_length = int(float(config.get("max_content_length", 1000.0)) * 1000.0)
        self.server.config["MAX_CONTENT_LENGTH"] = max_content_length
        self.max_models_per_gpu = max(1, config.get("max_models_per_gpu", 3))
        self.base_url = config.get("base_url", "")
        self.allow_origin = config.get("allow_origin", "*")
        self.timeout = float(config.get("timeout", 10.0))
        self.precision = config.get("precision", "fp32")
        logging.getLogger("werkzeug").disabled = True
        self.logger = get_logger("TEXT_CORRECTION_SERVER")
        self.logger.info(f"loaded server config:\n{pprint.pformat(config)}")
        self.num_gpus = torch.cuda.device_count()

        @self.server.after_request
        def _after_request(response: Response) -> Response:
            response.headers.add("Access-Control-Allow-Origin", server_allow_origin)
            response.headers.add("Access-Control-Allow-Private-Network", "true")
            return response

        @self.server.route(f"{server_base_url}/models")
        def _models() -> Response:
            response = jsonify([
                {
                    "task": text_corrector_cls.task,
                    "models": [
                        {"name": model.name, "description": model.description}
                        for model in text_corrector_cls.available_models()
                    ],
                }
                for text_corrector_cls in self.text_corrector_classes
            ])
            return response

        @self.server.route(f"{server_base_url}/info")
        def _info() -> Response:
            response = jsonify(
                {
                    "gpu": [gpu_info(i) for i in range(num_gpus)],
                    "cpu": cpu_info(),
                    "timeout": server_timeout,
                    "precision": server_precision,
                }
            )
            return response

        assert "models" in config and sum(len(names) for names in config["models"].values()) > 0, \
            "expected at least one model to be specified in the server config"
        self.text_correctors: Dict[Tuple[str, str], Tuple[TextCorrector, Optional[Stream], Lock]] = {}
        self.models_on_gpu: Dict[str, Set[Tuple[str, str]]] = {}
        self.models_in_use: Set[Tuple[str, str]] = set()
        self.placement_lock = Lock()
        for task, model_names in config["models"].items():
            text_corrector_cls: Optional[TextCorrector] = next(
                filter(lambda c: c.task == task, self.text_corrector_classes),
                None
            )
            assert text_corrector_cls is not None, \
                f"this server supports the tasks {[c.task for c in self.text_corrector_classes]}, but got unsupported task {task}"
            for model_name in model_names:
                text_corrector: TextCorrector = text_corrector_cls.from_pretrained(model_name, device)
                if num_gpus > 0:
                    device = f"cuda:{len(self.text_correctors) % num_gpus}"
                    stream = Stream(device)
                    if device not in self.models_on_gpu:
                        self.models_on_gpu[device] = set()
                    if len(self.models_on_gpu[device]) < self.max_models_per_gpu:
                        text_corrector = text_corrector.to(device)
                        self.models_on_gpu[device].add((task, model_name))
                else:
                    device = "cpu"
                    stream = None
                self.text_correctors[(task, model_name)] = (text_corrector, stream, device, Lock())

    def text_corrector(self, task: str, model_name: str) -> Union[TextCorrector, Tuple[str, int]]:
        if (task, model_name) not in self.text_correctors:
            return f"no model {model_name} for task {task} exists", 404
        cor, stream, device, lock = self.text_correctors[(task, model_name)]
        acquired = lock.acquire(timeout=self.timeout)
        if not acquired:
            return f"failed to reserve model within {self.timeout}s", 503
        acquired = self.placement_lock.acquire(timeout=self.timeout)
        if not acquired:
            lock.release()
            return f"failed to place model on correct device within {self.timeout}s", 503
        if device != "cpu" and (task, model_name) not in self.models_on_gpu:
            cor = cor.to(device)
        self.placement_lock.release()
        lock.release()

    def run(self):
        self.server.run("0.0.0.0", self.port, debug=False, use_reloader=False)
