import argparse
import logging
import sys
import os
from gate import Gate


def parse_args():
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
    parser.add_argument('--high_conf_sample_ratio', type=float, default=0.1, required=True, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, required=True, help='Batch size')
    parser.add_argument('--input_dim', type=int, default=1536, required=False, help='Input dimension')
    parser.add_argument('--embedding_dim', type=int, default=100, required=False,
                        help='Hidden layer embedding dimension')
    parser.add_argument('--ifIncrementalLearning', type=bool, default=False, help='If incrmental learning')
    parser.add_argument('--maxMLData', type=int, default=2000, help='the maximum number of data inserted into training data')
    parser.add_argument('--conf_threshold', type=float, default=0.7, help='Confidence threshold')
    #  variants of GATE: gate, creator, critic, creatornc, creatorne, creatorna, gatenc and creatoritr
    parser.add_argument('--variants', type=str, default='gate', help='Variants of GATE')

    # varuing the number of \Gamma
    parser.add_argument('--gamma', type=float, default=1.0, help='The ratio of Gamma')
    # varying the number of entity ids
    parser.add_argument('--entityRatio', type=float, default=1.0, help='The ratio of Entities')
    # varying CCs
    parser.add_argument('--ccs', type=str, default='CCs.txt', help='CCs file')
    # set gpu
    parser.add_argument('--gpuOption', type=str, default='0', help='set cuda gpu')
    # set |D_T|
    parser.add_argument('--D_T', type=float, default=1.0, help='The ratio of D_T')

    args = parser.parse_args() #(convert_arg_line_to_args(open(sys.argv[1]).read()))
    return args


def log_config(args):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s\t%(message)s', datefmt='%I:%M:%S',
        level=logging.INFO,
        filename='results/' + args.data.split('/')[-1] + '_' + str(args.batch_size) + '_' + str(args.epoch) + '_' + str(args.high_conf_sample_ratio) + '.log',
        filemode='a'
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
    #args = parse_args(sys.argv[1])
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuOption

    #log_config(args)
    gate = Gate()
    gate.initialize(args)
    #gate.train()
    gate.train_()
    #gate.evaluate()
