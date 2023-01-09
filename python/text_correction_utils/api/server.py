import os
from contextlib import contextmanager
import pprint
import logging
import json
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple, Type, Union, Generator

import torch
from torch.cuda import Stream
from flask import Flask, Response, cli, jsonify

from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils.api.utils import gpu_info, cpu_info
from text_correction_utils.logging import get_logger


class Error:
    def __init__(self, msg: str, status: int):
        self.msg = msg
        self.status = status

    def to_response(self) -> Response:
        return Response(self.msg, status=self.status)


class TextCorrectionServer:
    text_corrector_classes: List[Type[TextCorrector]]

    @staticmethod
    def from_config(path: str) -> "TextCorrectionServer":
        with open(path, "r", encoding="utf8") as inf:
            config = json.loads(inf.read())
        return TextCorrectionServer(config)

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port = int(self.config.get("port", 40000))
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
            response.headers.add("Access-Control-Allow-Origin", self.allow_origin)
            response.headers.add("Access-Control-Allow-Private-Network", "true")
            return response

        @self.server.route(f"{self.base_url}/models")
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

        @self.server.route(f"{self.base_url}/info")
        def _info() -> Response:
            response = jsonify(
                {
                    "gpu": [gpu_info(i) for i in range(self.num_gpus)],
                    "cpu": cpu_info(),
                    "timeout": self.timeout,
                    "precision": self.precision,
                }
            )
            return response

        assert "models" in config and sum(len(names) for names in config["models"].values()) > 0, \
            "expected at least one model to be specified in the server config"

        self.text_correctors: Dict[Tuple[str, str], Tuple[TextCorrector, Optional[Stream]]] = {}
        self.lock = Lock()
        for task, model_names in config["models"].items():
            text_corrector_cls: Optional[Type[TextCorrector]] = next(
                filter(lambda c: c.task == task, self.text_corrector_classes),
                None
            )
            assert text_corrector_cls is not None, \
                f"this server supports the tasks {[c.task for c in self.text_corrector_classes]}, \
but got unsupported task {task}"
            for model_name in model_names:
                if self.num_gpus > 0:
                    device = f"cuda:{len(self.text_correctors) % self.num_gpus}"
                    stream: Optional[Stream] = Stream(device)
                else:
                    device = "cpu"
                    stream = None
                text_corrector: TextCorrector = text_corrector_cls.from_pretrained(model_name, device)
                self.text_correctors[(task, model_name)] = (text_corrector, stream)

    @contextmanager
    def text_corrector(self, task: str, model_name: str) -> Generator[Union[TextCorrector, Error], None, None]:
        if (task, model_name) not in self.text_correctors:
            yield Error(f"no model {model_name} for task {task} exists", 404)
            return
        acquired = self.lock.acquire(timeout=self.timeout)
        if not acquired:
            yield Error(f"failed to reserve model within {self.timeout}s", 503)
            return
        cor, stream = self.text_correctors[(task, model_name)]
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    yield cor
            else:
                yield cor
        finally:
            self.lock.release()

    def run(self):
        self.server.run("0.0.0.0", self.port, debug=False, use_reloader=False)
