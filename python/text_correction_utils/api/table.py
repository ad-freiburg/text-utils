import os
import math
from typing import List, Optional, Set, Tuple

import torch
from torch import nn
from termcolor import colored

from text_correction_utils.api import utils


def generate_table(
    headers: List[List[str]],
    data: List[List[str]],
    alignments: Optional[List[str]] = None,
    horizontal_lines: Optional[List[int]] = None,
    highlight: Optional[Set[Tuple[int, int]]] = None,
    highlight_type: str = "markdown",
    highlight_color: str = "green",
    max_column_width: int = 48
) -> str:
    assert len(headers), "got no headers"
    assert len(set(len(header) for header in headers)) == 1, "all headers must have the same length"
    header_length = len(headers[0])

    assert all(header_length == len(item) for item in data), \
        f"header has length {header_length}, but data items have lengths {[len(item) for item in data]}"

    if alignments is None:
        alignments = ["left"] + ["right"] * (header_length - 1)

    if highlight is None:
        highlight = set()

    max_column_width = max(10, max_column_width)
    if len(highlight) > 0 and highlight_type == "markdown":
        max_column_width += 4

    # get max width for each column in headers and data
    column_widths = []
    for i in range(header_length):
        # add 4 to width if cell is bold because of the two **s left and right
        header_width = max(len(h[i]) for h in headers)
        data_width = max(
            min(
                max_column_width,
                len(d[i]) + (4 * ((j, i) in highlight and highlight_type == "markdown"))
            )
            for j, d in enumerate(data)
        )
        column_widths.append(
            min(
                max_column_width,
                max(
                    # markdown needs at least three - for a proper horizontal line
                    3,
                    header_width,
                    data_width
                )
            )
        )

    if horizontal_lines is None:
        horizontal_lines = [0] * len(data)

    highlight_cells = [
        [(i, j) in highlight for j in range(len(data[i]))]
        for i in range(len(data))
    ]

    tables_lines = []

    tables_lines.extend([
        _table_row(header, [False] * header_length, highlight_type,
                   highlight_color, alignments, column_widths, max_column_width)
        + (_table_horizontal_line(column_widths) if i == len(headers) - 1 else "")
        for i, header in enumerate(headers)
    ])

    for item, horizontal_line, bold in zip(data, horizontal_lines, highlight_cells):
        line = _table_row(item, bold, highlight_type, highlight_color,
                          alignments, column_widths, max_column_width)
        if horizontal_line > 0:
            line += _table_horizontal_line(column_widths)
        tables_lines.append(line)

    return "\n".join(tables_lines)


def _table_cell(s: str, alignment: str, width: int) -> str:
    if alignment == "left":
        s = s.ljust(width)
    elif alignment == "right":
        s = s.rjust(width)
    else:
        s = s.center(width)
    return s


def _highlight(s: str, hcolor: str) -> str:
    return colored(s, hcolor, attrs=["bold"])


def _table_row(
    data: List[str],
    highlight: List[bool],
    highlight_type: str,
    highlight_color: str,
    alignments: List[str],
    widths: List[int],
    max_width: int
) -> str:
    assert len(data) == len(highlight)
    num_lines = [math.ceil(len(d) / max_width) for d in data]
    max_num_lines = max(num_lines)
    lines = []
    for i in range(max_num_lines):
        line_data = [d[i*max_width: (i + 1) * max_width] for d in data]
        cells = []
        for d, h, a, w in zip(line_data, highlight, alignments, widths):
            if h and highlight_type == "markdown":
                cell = _table_cell(f"**{d}**", a, w)
            elif h and highlight_type == "terminal":
                cell = _highlight(_table_cell(d, a, w), highlight_color)
            else:
                cell = _table_cell(d, a, w)
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_horizontal_line(widths: List[int]) -> str:
    return "\n| " + " | ".join("-" * w for w in widths) + " |"


def generate_report(
        task: str,
        model_name: str,
        model: nn.Module,
        input_size: int,
        input_size_bytes: int,
        runtime: float,
        precision: torch.dtype,
        batch_size: int,
        sort_by_length: bool,
        device: torch.device,
        file_path: Optional[str] = None
) -> Optional[str]:
    if precision == torch.float16:
        precision_str = "fp16"
    elif precision == torch.bfloat16:
        precision_str = "bfp16"
    elif precision == torch.float32:
        precision_str = "fp32"
    else:
        raise ValueError("expected precision to be one of torch.float16, torch.bfloat16 or torch.float32")

    report = generate_table(
        headers=[["REPORT", task]],
        data=[
            ["Model", model_name],
            ["Input size 1", f"{input_size} sequences"],
            ["Input size 2", f"{input_size_bytes / 1000:,.2f} kB"],
            ["Runtime", f"{runtime:,.1f} s"],
            ["Throughput 1", f"{input_size / runtime:,.1f} Seq/s"],
            ["Throughput 2", f"{input_size_bytes / runtime / 1000:,.1f} kB/s"],
            ["GPU memory", f"{torch.cuda.max_memory_reserved(device) // 1024 ** 2:,} MiB"],
            ["Parameters", f"{utils.num_parameters(model)['total'] / 1000 ** 2:,.1f} M"],
            ["Precision", precision_str],
            ["Batch size", f"{batch_size:,}"],
            ["Sorted", "yes" if sort_by_length else "no"],
            ["Device",  f"{utils.cpu_info()}{', ' + utils.device_info(device) if device.type == 'cuda' else ''}"],
        ],
    )
    if file_path is not None:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf8") as of:
            of.write(report + "\n")

        return None
    else:
        return report
