import asyncio
from contextlib import asynccontextmanager
from typing import Any, Type

import torch
import uvicorn
import yaml
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from text_utils import configuration
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import cpu_info, gpu_info
from text_utils.logging import get_logger



class RequestCancelledMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Let's make a shared queue for the request messages
        queue = asyncio.Queue()

        async def message_poller(sentinel: object, handler_task: asyncio.Task):
            nonlocal queue
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    handler_task.cancel()
                    return sentinel  # Break the loop

                # Puts the message in the queue
                await queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, queue.get, send))  # type: ignore
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            return await handler_task
        except asyncio.CancelledError:
            pass


class Error:
    def __init__(self, error: str, status_code: int):
        self.error = error
        self.status_code = status_code

    def to_response(self) -> JSONResponse:
        return JSONResponse({"error": self.error}, self.status_code)


class TextProcessingServer:
    text_processor_cls: Type[TextProcessor]

    @classmethod
    def from_config(
        cls, path: str, log_level: str | int | None = None
    ) -> "TextProcessingServer":
        config = configuration.load_config(path)
        return cls(config, log_level)

    def __init__(self, config: dict[str, Any], log_level: str | int | None = None):
        self.config = config
        self.logger = get_logger(
            f"{self.text_processor_cls.task.upper()} SERVER",
            log_level,
        )
        self.logger.info(f"Loaded server config:\n{yaml.dump(config)}")
        self.port = int(self.config.get("port", 40000))

        self.server = FastAPI()
        self.server.add_middleware(
            CORSMiddleware,
            allow_origins=[config.get("allow_origin", "*")],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        self.server.add_middleware(RequestCancelledMiddleware)

        self.max_models_per_gpu = max(1, config.get("max_models_per_gpu", 1))
        self.timeout = config.get("timeout", 10.0)
        self.num_gpus = torch.cuda.device_count()

        assert (
            "models" in config and len(config["models"]) > 0
        ), "Expected at least one model to be specified in the server config"

        self.text_processors: dict[str, TextProcessor] = {}
        self.model_infos = {}
        self.model_cfgs = {}
        assert "models" in config, "Expected models in server config"
        for name, model_cfg in config["models"].items():
            if "device" in model_cfg:
                device = model_cfg["device"]
            elif self.num_gpus > 0:
                device = f"cuda:{len(self.text_processors) % self.num_gpus}"
            else:
                device = "cpu"

            if "name" in model_cfg:
                model_name = model_cfg["name"]
                model_info = next(
                    filter(
                        lambda m: m.name == model_name,
                        self.text_processor_cls.available_models(),
                    ),
                    None,
                )
                if model_info is None:
                    raise RuntimeError(
                        f"Model {model_name} not found in available models"
                    )
                self.logger.info(
                    f"Loading pretrained model {model_info.name} for task "
                    f"{self.text_processor_cls.task} onto device {device}"
                )
                text_processor = self.text_processor_cls.from_pretrained(
                    model_name, device
                )
                model_info.tags.append("src::pretrained")

            elif "path" in model_cfg:
                path = model_cfg["path"]
                self.logger.info(
                    f"Loading model for task {self.text_processor_cls.task} "
                    f"from experiment {path} onto device {device}"
                )
                text_processor = self.text_processor_cls.from_experiment(path, device)

                model_info = ModelInfo(
                    name=text_processor.name,
                    description="Loaded from custom experiment",
                    tags=["src::experiment"],
                )

            else:
                raise RuntimeError("Expected either name or path in model config")

            # handle the case when two models have the same name
            if name in self.text_processors:
                raise RuntimeError(f"Got multiple models with name '{name}'")

            self.model_infos[name] = model_info
            self.model_cfgs[name] = model_cfg
            self.text_processors[name] = text_processor

        self.lock = asyncio.Lock()

        @self.server.get("/info")
        async def info() -> dict[str, Any]:
            return {
                "gpu": [gpu_info(i) for i in range(self.num_gpus)],
                "cpu": cpu_info(),
                "timeout": self.timeout,
            }

        @self.server.get("/models")
        async def models() -> dict[str, Any]:
            return {
                "task": self.text_processor_cls.task,
                "models": {
                    name: info._asdict() for name, info in self.model_infos.items()
                },
            }

    @asynccontextmanager
    async def get_text_processor(self, name: str):
        if name not in self.text_processors:
            yield Error(f"Model {name} does not exist", status.HTTP_404_NOT_FOUND)
            return

        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=self.timeout)

            yield self.text_processors[name]

        except asyncio.TimeoutError:
            yield Error(
                f"Failed to acquire lock within {self.timeout:.2f}s",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        finally:
            if self.lock.locked():
                self.lock.release()

    def run(self):
        uvicorn.run(
            self.server,
            host="0.0.0.0",
            port=self.port,
            log_level=self.logger.level,
            limit_concurrency=32,
        )
