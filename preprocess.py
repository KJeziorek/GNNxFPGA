from data.ncaltech101 import NCaltech101
from data.ncars import NCars

import argparse
import multiprocessing as mp


def main(args):
    # dm = NCaltech101(data_dir='dataset', batch_size=1)
    dm = NCars(data_dir='dataset', batch_size=1)
    dm.prepare_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--data_name', type=str, default='ncaltech101')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    main(args)