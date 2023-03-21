import logging
from os import PathLike
from pathlib import Path

from requests import Session
from tqdm import tqdm

try:
    from .version import __version__
except ImportError:
    __version__ = '0.0.0'

CHUNK_SIZE = 4096

LLAMA_COMMON_FILES = ['tokenizer_checklist.chk', 'tokenizer.model']
LLAMA_MODEL_FILES = ['checklist.chk', 'params.json']
LLAMA_NOPARTS = {'7B': 1, '13B': 2, '30B': 4, '65B': 8}
LLAMA_URL = 'https://agi.gpt4.org/llama/LLaMA/{endpoint}'


def fetch(targets, model_dir: Path, sess=None):
    sess = sess or Session()
    sess = Session()
    sess.headers['user-agent'] = f'llama.py/{__version__}'
    sess.stream = True
    for target in targets:
        path = model_dir / target
        logging.info('fetch %s to %s', target, path)

        url = LLAMA_URL.format(endpoint=target)
        res = sess.get(url)

        try:
            total = int(res.headers.get('content-length', ''))
        except ValueError:
            total = None

        with open(path, 'wb') as fout:
            with tqdm(desc=target, total=total, leave=False, unit='b',
                      unit_divisor=1024, unit_scale=True) as pbar:
                for chunk in res.iter_content(CHUNK_SIZE):
                    if chunk:
                        fout.write(chunk)
                        pbar.update(len(chunk))


def pull(name: str, size: str, outdir: PathLike):
    """Fetch model checkpoints, vocabulary, and other metadata and stores it on
    filesystem.

    Args:
      name: Model name (e.g. llama).
      size: Model size variant (e.g. 7B or 65B).
      outdir: Output directory to store files.
    """
    model_dir = Path(outdir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if name == 'llama':
        pull_llama(size, model_dir)
    else:
        raise ValueError(f'Unknown model: {name}.')


def pull_llama(size: str, model_dir: Path):
    if size not in LLAMA_NOPARTS:
        raise ValueError(f'Unknown model size: {size}. Available model sizes '
                         f'are {LLAMA_NOPARTS.values()}.')

    model_variant_dir = model_dir / size
    model_variant_dir.mkdir(exist_ok=True)

    targets = LLAMA_COMMON_FILES + [f'{size}/{x}' for x in LLAMA_MODEL_FILES]
    for part in range(LLAMA_NOPARTS[size]):
        targets.append(f'{size}/consolidated.{part:02d}.pth')
    logging.info('total %d files to download', len(targets))

    fetch(targets, model_dir)
