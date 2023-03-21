import inspect
import logging
from argparse import ArgumentParser, FileType
from pathlib import Path
from sys import stderr

try:
    from .version import __version__
except ImportError:
    __version__ = '0.0.0'

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARN,
    'error': logging.ERROR,
}


def convert(fp_type: int, model_dir: Path):
    from .io import convert
    convert(model_dir, fp_type)


def help_():
    parser.print_help()


def pull(model_dir: Path, model_size: str):
    from .zoo import pull
    pull('llama', model_size, model_dir)


def serve():
    pass


def version_():
    print(f'llama.py version {__version__}')


def main():
    # Parse command line arguments. If no subcommand were run then show usage
    # and exit. We assume that only main parser (super command) has valid value
    # in func attribute.
    args = parser.parse_args()
    if args.func is None:
        parser.print_usage()
        return

    # Find CLI option or argument by parameter name of handling function.
    kwargs = {}
    spec = inspect.getfullargspec(args.func)
    for name in spec.args:
        kwargs[name] = getattr(args, name)

    # Set up basic logging configuration.
    if (stream := args.log_output) is None:
        stream = stderr

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=LOG_LEVELS[args.log_level],
                        stream=stream)

    # Invoke CLI command handler.
    args.func(**kwargs)


# Parser for connectivity options.
parser_opt_connection = ArgumentParser(add_help=False)
parser_opt_connection_group = parser_opt_connection.add_argument_group('connection options')  # noqa: E501
parser_opt_connection_group.add_argument('-H', '--host', type=str, help='Address to listen.')  # noqa: E501
parser_opt_connection_group.add_argument('-p', '--port', type=int, help='Port to listen.')  # noqa: E501

# Parser for model options.
parser_opt_model = ArgumentParser(add_help=False)
parser_opt_model_group = parser_opt_model.add_argument_group('model options')

# Root parser for the tool.
parser = ArgumentParser(description=__doc__)
parser.set_defaults(func=None)
parser.add_argument('--log-level', default='info', choices=sorted(LOG_LEVELS.keys()), help='set logger verbosity level')  # noqa: E501
parser.add_argument('--log-output', default=stderr, metavar='FILENAME', type=FileType('w'), help='set output file or stderr (-) for logging')  # noqa: E501

subparsers = parser.add_subparsers()

parser_convert = subparsers.add_parser('convert', help='convert model from pth to gglm format')  # noqa: E501
parser_convert.set_defaults(func=convert)
parser_convert.add_argument('--fp-type', choices=('fp16', 'fp32'), default='fp16', help='target floating point type')  # noqa: E501
parser_convert.add_argument('model_dir', type=Path, default=Path('.'), help='model directory')  # noqa: E501

parser_help = subparsers.add_parser('help', add_help=False, help='show this message and exit')  # noqa: E501
parser_help.set_defaults(func=help_)

parser_pull = subparsers.add_parser('pull', help='download model')  # noqa: E501
parser_pull.set_defaults(func=pull)
parser_pull.add_argument('-m', '--model-dir', type=Path, default=Path('.'), help='download directory ')  # noqa: E501
parser_pull.add_argument('-s', '--model-size', choices=('7B', '13B', '30B', '65B'), default='7B', help='model variant')  # noqa: E501

parser_serve = subparsers.add_parser('serve', parents=[parser_opt_connection, parser_opt_model], help='run language server')  # noqa: E501
parser_serve.set_defaults(func=serve)

parser_version = subparsers.add_parser('version', add_help=False, help='show version and exit')  # noqa: E501
parser_version.set_defaults(func=version_)
