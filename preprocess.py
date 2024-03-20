from data.ncaltech101 import NCaltech101
from data.ncars import NCars

import argparse
import multiprocessing as mp


def main(args):
    dm = NCaltech101(data_dir='dataset', batch_size=1, radius=3)
    # dm = NCars(data_dir='dataset', batch_size=1, radius=3)
    dm.prepare_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    main(args)