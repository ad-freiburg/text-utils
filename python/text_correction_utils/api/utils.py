import logging
import os
import platform
import re
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Union, Dict, List, Any, Optional, Iterator, Callable

import requests
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


from text_correction_utils import data


def _unpack_zip(
        zip_file_path: str,
        directory: str
) -> None:
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(directory)


def to(v: Any, device: torch.device) -> Any:
    if isinstance(v, torch.Tensor):
        return v.to(device, non_blocking=True)
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v).to(device, non_blocking=True)
    elif isinstance(v, dict):
        return {k: to(v_, device) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [to(v_, device) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(to(v_, device) for v_ in v)
    else:
        return v


class ProgressIterator:
    # a utility class capturing the number and total size
    # of items passed through an iterator
    def __init__(
        self,
        it: Iterator[Any],
        size_fn: Callable[[Any], int]
    ):
        self.it = it
        self.num_items = 0
        self.total_size = 0
        self.size_fn = size_fn

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.it)
        self.num_items += 1
        self.total_size += self.size_fn(item)
        return item


def download_zip(
        name: str,
        url: str,
        download_dir: str,
        cache_dir: str,
        sub_cache_dir: str,
        force_download: bool,
        logger: logging.Logger
) -> str:
    """
    Downloads and extracts a zip into cache dir and returns the path to the only subdirectory
    :param name: informative name of the zip file content
    :param url: url of the zip
    :param download_dir: directory to store zipped content
    :param cache_dir: directory to cache unzipped content
    :param sub_cache_dir: directory relative to the cache_dir where zip will be extracted to
    :param force_download: download even if it is already in the cache dir
    :param logger: instance of a logger to log some useful information
    :return: path of to unzipped subdirectory
    """
    zip_file_path = os.path.join(download_dir, url.split("/")[-1])
    not_downloaded = not os.path.exists(zip_file_path)
    if not_downloaded or force_download:
        directory = os.path.dirname(zip_file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        logger.info(f"downloading {name} from {url} to download directory {download_dir}")
        response = requests.get(url, stream=True)
        if not response.ok:
            raise RuntimeError(f"error downloading {name} from {url}: "
                               f"status {response.status_code}, {response.reason}")

        try:
            file_size = int(response.headers.get("content-length", 0))
            pbar = byte_progress_bar(f"downloading {name}", file_size)

            with open(zip_file_path, "wb") as of:
                for data in response.iter_content():
                    of.write(data)
                    pbar.update(len(data))

            pbar.close()

        except BaseException as e:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            raise e
    else:
        logger.info(f"{name} is already downloaded to download directory {download_dir}")

    zip_dir = os.path.join(cache_dir, sub_cache_dir)
    not_extracted = not os.path.exists(zip_dir)
    if not_extracted or force_download:
        shutil.rmtree(zip_dir, ignore_errors=True)
        _unpack_zip(zip_file_path, zip_dir)
    return zip_dir


def cpu_info() -> str:
    if platform.system() == "Linux":
        try:
            cpu_regex = re.compile(r"model name\t: (.*)", re.DOTALL)
            with open("/proc/cpuinfo", "r", encoding="utf8") as inf:
                cpu_info = inf.readlines()

            for line in cpu_info:
                line = line.strip()
                match = cpu_regex.match(line)
                if match is not None:
                    return match.group(1)
        except BaseException:
            return platform.processor()
    return platform.processor()


def gpu_info(device: Union[torch.device, str, int]) -> str:
    device_props = torch.cuda.get_device_properties(device)
    return f"{device_props.name} ({device_props.total_memory // 1024 // 1024:,}MiB memory, " \
           f"{device_props.major}.{device_props.minor} compute capability, " \
           f"{device_props.multi_processor_count} multiprocessors)"


def device_info(device: torch.device) -> str:
    return gpu_info(device) if device.type == "cuda" else cpu_info()


def _run_cmd(path: str, cmd: List[str]) -> str:
    return subprocess.check_output(
        cmd,
        cwd=Path(path).resolve()
    ).strip().decode("utf8")


def git_branch(path: str) -> str:
    return _run_cmd(path, ["git", "branch", "--show-current"])


def git_commit(path: str) -> str:
    return _run_cmd(path, ["git", "rev-parse", "HEAD"])


def num_parameters(module: nn.Module) -> Dict[str, int]:
    """

    Get the number of trainable, fixed and total parameters of a pytorch module.

    :param module: pytorch module
    :return: dict containing number of parameters
    """
    trainable = 0
    fixed = 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            fixed += p.numel()
    return {"trainable": trainable, "fixed": fixed, "total": trainable + fixed}


def sequence_progress_bar(desc: str, total: int, disable: bool = False) -> tqdm:
    return tqdm(
        desc=desc,
        total=total,
        disable=disable,
        ascii=True,
        leave=False,
        unit="Seq",
        unit_scale=True,
        unit_divisor=1000
    )


def byte_progress_bar(desc: str, total: Optional[int] = None, disable: bool = False) -> tqdm:
    return tqdm(
        desc=desc,
        total=total,
        disable=disable,
        ascii=True,
        leave=False,
        unit="B",
        unit_scale=True,
        unit_divisor=1000
    )
