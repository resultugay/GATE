# Import the library
import argparse
import os
from Gate import Gate
import logging

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description='GATE',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--dataset', type=str, default=None, required=False, help='Specify dataset')
    parser.add_argument('--columns', type=str, default=None, required=True, help='Specify dataset')
    parser.add_argument('--creator', type=str, default=None, required=True, help='Specify creator')
    parser.add_argument('--critic', type=str, default=None, required=True, help='Specify critic')
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
    gate = Gate()
    gate.initialize(args)
    gate.train()
    gate.evaluate()
