import json
from typing import Dict, Any


class TextCorrectionServer:
    @staticmethod
    def from_config(path: str) -> "TextCorrectionServer":
        with open(path, "r", encoding="utf8") as inf:
            config = json.loads(inf.read())
        return TextCorrectionServer(config)

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self):
        raise NotImplementedError
