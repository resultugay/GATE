import argparse
import logging
import sys
from gate import Gate


def parse_args(path):
    parser = argparse.ArgumentParser(
        description='GATE',
        usage='main.py [<args>] [-h | --help]',
        fromfile_prefix_chars='@'
    )
    parser.add_argument('--data', type=str, default=None, required=True, help='Specify path of the data')
    parser.add_argument('--creator', type=str, default='gate', required=True, help='Specify creator')
    parser.add_argument('--critic', type=str, default=None, required=False, help='Specify critic')
    parser.add_argument('--epoch', type=int, default=100, required=True, help='Epoch Number')
    parser.add_argument('--lr', type=float, default=0.001, required=True, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, required=True, help='Batch size')
    parser.add_argument('--input_dim', type=int, default=1536, required=False, help='Input dimension')
    parser.add_argument('--embedding_dim', type=int, default=64, required=False,
                        help='Hidden layer embedding dimension')

    args = parser.parse_args(convert_arg_line_to_args(open(sys.argv[1]).read()))
    return args


def log_config():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s\t%(message)s', datefmt='%I:%M:%S',
        level=logging.INFO,
        # filename='example.log',
    )


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logging.error("Please provide the config file")
        sys.exit(0)
    args = parse_args(sys.argv[1])

    log_config()
    gate = Gate()
    gate.initialize(args)
    gate.train()
