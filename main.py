import argparse
import logging
import sys

from Gate import Gate


def parse_args(path):
    parser = argparse.ArgumentParser(
        description='GATE',
        usage='main.py [<args>] [-h | --help]',
        fromfile_prefix_chars='@'
    )
    parser.add_argument('--training_data', type=str, default=None, required=True, help='Specify training data')
    parser.add_argument('--dataset', type=str, default=None, required=True, help='Specify dataset name')
    parser.add_argument('--test_data', type=str, default=None, required=True, help='Specify test data')
    parser.add_argument('--temporal_columns', type=str, default=None,nargs='+', required=True, help='Specify temporal columns')
    parser.add_argument('--non_temporal_columns', type=str, default=None,nargs='+', required=True, help='Specify non temporal columns')
    parser.add_argument('--creator', type=str, default='AV2', required=True, help='Specify creator')
    parser.add_argument('--critic', type=str, default='Denial', required=True, help='Specify critic')
    parser.add_argument('--epoch', type=int, default=100, required=True, help='Epoch Number')
    parser.add_argument('--emb_dim', type=int, default=2, required=True, help='Embedding dimension of attribute values')
    parser.add_argument('--write_every', type=int, default=1, required=True, help='Write output of the epoch at every iteration')
    parser.add_argument('--lr', type=float, default=0.001, required=True, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=1, required=True, help='Batch Size')
    parser.add_argument('--training', type=int, default=1, required=True, help='Training ?')
    parser.add_argument('--test', type=int, default=1, required=True, help='Test ?')
    parser.add_argument('--word_embedding', type=str, required=False, help='Word Embedding file')

    """ # using shlex parse arguments
    import shlex
    if sys.argv[1].startswith('@'):
        print(sys.argv[1][1:])
        args = parser.parse_args(shlex.split(open(sys.argv[1][1:]).read()))
        return args
    """
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
    gate.evaluate()
