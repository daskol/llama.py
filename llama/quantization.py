import logging
import re
from os import PathLike
from pathlib import Path

from ._llama import GGMLType, quantize_model

RE_CHECKPOINT = re.compile(r'ggml-model-(f16|f32).bin(.(\d\d))?')


def quantize(model_dir: PathLike, q_type='q4_0'):
    q_type_code = GGMLType.Q4_0  # TODO: Should not be hardcoded.
    for path in Path(model_dir).iterdir():
        if (m := RE_CHECKPOINT.match(path.name)) is None:
            continue

        logging.info('quantize model checkpoint %s', path)
        fp_type = m.group(1)
        filename = path.name.replace(fp_type, q_type)
        quantize_model(str(path), str(path.with_name(filename)), q_type_code)
