import logging
import re
from json import dumps, load
from os import PathLike
from pathlib import Path
from struct import pack

import numpy as np
import torch as T
from sentencepiece import SentencePieceProcessor

GGLM_HEADER_KEYS = ['vocab_size', 'dim', 'multiple_of', 'n_heads', 'n_layers']
GGLM_MAGIC = 0x67676d66
GGLM_VERSION = 1

RE_CHECKPOINT = re.compile(r'consolidated.(\d\d).pth')


def convert(model_dir: PathLike, fp_type: str = 'fp16'):
    """Convert model located at :model_dir: to gglm format.

    Args:
        model_dir: Directory where model is located.
        fp_type: Target FP-type of weights.
    """
    if fp_type not in ('fp16', 'fp32'):
        raise ValueError('Param fp_type should either fp16 or fp32.')

    model_dir = Path(model_dir)
    hparams, tokenizer = load_hparams_and_tokenizer(model_dir)
    logging.info('model hyperparams: %s', dumps(hparams))

    def convert_partition(src, dst):
        logging.info('convert %s to %s', src, dst)
        model = T.load(src, map_location='cpu')
        with open(dst, 'wb') as fout:
            write_gglm_header(hparams, fout, fp_type)
            write_gglm_tokens(tokenizer, fout)
            write_gglm_body(model, fout, fp_type)

    for path in model_dir.iterdir():
        if (m := RE_CHECKPOINT.match(path.name)) is None:
            continue

        index = int(m.group(1))
        suffix = str(index) if index else ''
        filename = f'ggml-model-{fp_type}.bin{suffix}'
        convert_partition(path, path.with_name(filename))


def load_hparams_and_tokenizer(model_dir: PathLike):
    model_dir = Path(model_dir)
    model_parent_dir = model_dir / '..'
    with open(model_dir / 'params.json') as fin:
        hparams = load(fin)
    tokenizer_path = str(model_parent_dir / 'tokenizer.model')
    tokenizer = SentencePieceProcessor(tokenizer_path)
    hparams.update({"vocab_size": tokenizer.vocab_size()})
    return hparams, tokenizer


def write_gglm_body(state_dict, fout, ftype='fp16'):
    for key, val in state_dict.items():
        if key.endswith('freqs'):
            continue

        logging.info('processing variable %s: shape=%s dtype=%s', key,
                     val.shape, val.dtype)

        # Default type is fp16.
        ftype_cur = 'fp16'
        if ftype == 'fp32' or val.ndim == 1:
            logging.info('  converting to float32')
            data = data.astype(np.float32)
            ftype_cur = 'fp32'

        ftype_code = 0
        if ftype_cur == 'fp16':
            ftype_code = 1

        # Write header (sizes of shape, length of key, fp-type, shape, key).
        key_encoded = key.encode('utf-8')
        data = val.numpy().squeeze()
        fout.write(pack('3i', len(data.shape), len(key_encoded), ftype_code))
        for dim in reversed(data.shape):
            fout.write(pack('i', dim))
        fout.write(key_encoded)

        # Write floating point numbers.
        data.tofile(fout)


def write_gglm_header(hparams, fout, fp_type: str):
    fp_code = 0
    if fp_type == 'fp16':
        fp_code = 1
    values = [
        GGLM_MAGIC,
        GGLM_VERSION,
        *[hparams[key] for key in GGLM_HEADER_KEYS],
        hparams['dim'] // hparams['n_heads'],  # Obsolete.
        fp_code,
    ]
    fout.write(pack('i' * len(values), *values))


def write_gglm_tokens(tokenizer, fout):
    for token_id in range(tokenizer.vocab_size()):
        if tokenizer.is_unknown(token_id):
            text = ' \u2047 '.encode('utf-8')
        elif tokenizer.is_control(token_id):
            text = b''
        elif tokenizer.is_byte(token_id):
            token = tokenizer.id_to_piece(token_id)
            if len(token) != 6:
                raise RuntimeError(f'Invalid token #{token_id}: `{token}`.')
            byte_value = int(token[3:-1], 16)
            text = pack('B', byte_value)
        else:
            token = tokenizer.id_to_piece(token_id)
            text = token.replace('\u2581', ' ').encode('utf-8')
        fout.write(pack('i', len(text)))
        fout.write(text)
        fout.write(pack('f', tokenizer.get_score(token_id)))
