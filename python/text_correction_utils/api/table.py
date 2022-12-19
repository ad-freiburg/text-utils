import os
from typing import List, Optional, Set, Tuple

import torch

from text_correction_utils.api import utils


def generate_table(
    headers: List[List[str]],
    data: List[List[str]],
    alignments: Optional[List[str]] = None,
    horizontal_lines: Optional[List[int]] = None,
    bold_cells: Optional[Set[Tuple[int, int]]] = None,
    vertical_lines: bool = False,
    fmt: str = "markdown"
) -> str:
    assert fmt in {"markdown", "latex"}

    assert len(headers), "got no headers"
    assert len(set(len(header) for header in headers)) == 1, "all headers must have the same length"
    header_length = len(headers[0])

    assert all(header_length == len(item) for item in data), \
        f"header has length {header_length}, but data items have lengths {[len(item) for item in data]}"

    if alignments is None:
        alignments = ["left"] + ["right"] * (header_length - 1)

    if horizontal_lines is None or fmt == "markdown":
        horizontal_lines = [0] * len(data)
    horizontal_lines[-1] = fmt == "latex"  # always a horizontal line after last line for latex, but not for markdown

    if bold_cells is None:
        bold_cells = [[False] * len(item) for item in data]
    else:
        bold_cells = [[(i, j) in bold_cells for j in range(len(data[i]))] for i in range(len(data))]

    tables_lines = []

    opening_str = _open_table(fmt, alignments, vertical_lines)
    if opening_str:
        tables_lines.append(opening_str)

    tables_lines.extend([
        _table_row(fmt, header, [False] * header_length)
        + (_table_horizontal_line(fmt, header_length, alignments, 2) if i == len(headers) - 1 else "")
        for i, header in enumerate(headers)
    ])

    for item, horizontal_line, bold in zip(data, horizontal_lines, bold_cells):
        line = _table_row(fmt, item, bold)
        if horizontal_line > 0:
            line += _table_horizontal_line(fmt, len(item), alignments, horizontal_line)
        tables_lines.append(line)

    closing_str = _close_table(fmt)
    if closing_str:
        tables_lines.append(closing_str)

    return "\n".join(tables_lines)


_MARK_DOWN_ALIGNMENTS = {
    "center": ":---:",
    "left": ":--",
    "right": "--:"
}

_LATEX_ALIGNMENTS = {
    "center": "c",
    "left": "l",
    "right": "r"
}


def _open_table(fmt: str, alignments: List[str], vertical_lines: bool) -> str:
    if fmt == "markdown":
        return ""
    else:
        divider = "|" if vertical_lines else ""
        return f"\\begin{{tabular}}{{{divider}" \
               + f"{divider}".join(_LATEX_ALIGNMENTS[align] for align in alignments) \
               + f"{divider}}} \\hline"


def _close_table(fmt: str) -> str:
    if fmt == "markdown":
        return ""
    else:
        return "\\end{tabular}"


_LATEX_ESCAPE_CHARS = {"_", "%"}  # "&", "%", "$", "#", "_", "{", "}"}


def _format_latex(s: str, bold: bool) -> str:
    s = "".join("\\" + char if char in _LATEX_ESCAPE_CHARS else char for char in s)
    if bold:
        s = "\\textbf{" + s + "}"
    return s


def _format_markdown(s: str, bold: bool) -> str:
    if bold:
        s = "**" + s + "**"
    return s


def _table_row(fmt: str, data: List[str], bold: List[bool]) -> str:
    assert len(data) == len(bold)

    if fmt == "markdown":
        return "| " + " | ".join(_format_markdown(s, b) for s, b in zip(data, bold)) + " |"
    else:
        return " & ".join(_format_latex(s, b) for s, b in zip(data, bold)) + " \\\\ "


def _table_horizontal_line(fmt: str, num_cols: int, alignments: List[str], num_lines: int) -> str:
    assert num_cols == len(alignments) and all(align in {"left", "right", "center"} for align in alignments)

    if fmt == "markdown":
        return "\n| " + " | ".join(_MARK_DOWN_ALIGNMENTS[align] for align in alignments) + " |"
    else:
        assert num_lines in {1, 2}
        return "\\hline" * num_lines


def generate_report(
        task: str,
        model_name: str,
        model: nn.Module,
        inputs: List[str],
        runtime: float,
        precision: torch.dtype,
        batch_size: int,
        sort_by_length: bool,
        device: torch.device,
        file_path: Optional[str] = None
) -> Optional[str]:
    input_size = len(inputs)
    input_size_bytes = sum(len(ipt.encode("utf8")) for ipt in inputs)

    if precision == torch.float16:
        precision_str = "fp16"
    elif precision == torch.bfloat16:
        precision_str = "bfp16"
    elif precision == torch.float32:
        precision_str = "fp32"
    else:
        raise ValueError("expected precision to be one of torch.float16, torch.bfloat16 or torch.float32")

    report = generate_table(
        headers=[[
            "Task",
            "Model",
            "Input size",
            "Runtime in seconds",
            "Seq/s",
            "kB/s",
            "MiB GPU memory",
            "Mio. parameters",
            "Precision",
            "Batch size",
            "Sorted",
            "Device"
        ]],
        data=[
            [
                task,
                model_name,
                f"{input_size:,} seq, {input_size_bytes / 1000:,.2f} kB",
                f"{runtime:.1f}",
                f"{input_size / runtime:.1f}",
                f"{input_size_bytes / (runtime * 1000):.1f}",
                f"{torch.cuda.max_memory_reserved(device) // (1024 ** 2):,}" if device.type == "cuda" else "-",
                f"{utils.num_parameters(model)['total'] / (1000 ** 2):,.1f}",
                precision_str,
                str(batch_size),
                "yes" if sort_by_length else "no",
                f"{torch.cuda.get_device_name(device)}, {utils.device_info(device)}"
            ]
        ],
        fmt="markdown"
    )
    if file_path is not None:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        exists = os.path.exists(file_path)
        with open(file_path, "a" if exists else "w", encoding="utf8") as of:
            if exists:
                of.write(report.splitlines()[-1] + "\n")
            else:
                of.write(report + "\n")
        return None
    else:
        return report
