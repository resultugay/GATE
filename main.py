# Import the library
import argparse
import logging
import os

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='GATE',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--dataset', type=str, default=None, required=True, help='Specify dataset')
    parser.add_argument('--word_embedding', type=str, required=False, help='Word Embedding file')
    return parser.parse_args()


def log_config():
    logging.basicConfig(
                format='%(asctime)s %(levelname)s\t%(message)s', datefmt='%I:%M:%S',
                level=logging.INFO,
                #filename='example.log',
            )

if __name__ == '__main__':
    args = parse_args()
    log_config()
    logging.info('GATE Started')
