from text_correction_utils.version import __version__
from ._internal import (
    edit,
    text,
    whitespace,
    data,
    tokenization,
    dictionary,
    windows,
    metrics,
    unicode
)
from text_correction_utils import (
    api,
    inference,
    modules,
    logging,
    distributed,
    mask,
    io
)
import sys
import os
sys.path.append(os.path.dirname(__file__))
