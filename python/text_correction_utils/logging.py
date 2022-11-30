import logging

_LOG_FORMAT = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"

__all__ = ["setup_logging", "add_file_log", "get_logger", "eta_minutes_message", "eta_seconds_message"]


def setup_logging(level: int = logging.INFO) -> None:
    """

    Sets up logging with a custom log format and level.

    :param level: log level
    :return: None
    """
    logging.basicConfig(format=_LOG_FORMAT)
    logging.getLogger().setLevel(level)


def add_file_log(logger: logging.Logger, log_file: str) -> None:
    """

    Add file logging to an existing logger

    :param logger: logger
    :param log_file: path to logfile
    :return: logger with file logging handler
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(file_handler)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """

    Get a logger that writes to stderr.

    :param name: name of the logger
    :param level: log level
    :return: logger
    """

    logger = logging.getLogger(name)
    logger.propagate = False
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    if not logger.hasHandlers():
        logger.addHandler(stderr_handler)
    logger.setLevel(level)
    return logger


def eta(dur: float, num_iter: int, total_iter: int) -> float:
    """

    Calculate remaining time to reach total_iter iterations.

    :param dur: time spent for num_iter iterations
    :param num_iter: number of iterations so far
    :param total_iter: number of total iterations
    :return: time remaining

    >>> eta(10, 2, 10)
    40.0

    """
    return (dur / num_iter) * total_iter - dur


def eta_minutes_message(num_minutes: float, num_iter: int, total_iter: int) -> str:
    eta_minutes = eta(num_minutes, num_iter, total_iter)
    return f"{num_minutes:.2f} minutes since start, {eta_minutes:.2f} minutes to go"


def eta_seconds_message(num_sec: float, num_iter: int, total_iter: int) -> str:
    eta_seconds = eta(num_sec, num_iter, total_iter)
    return f"{num_sec:.2f} seconds since start, {eta_seconds:.2f} seconds to go"
