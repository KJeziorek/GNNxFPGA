from data.data_module import EventDM

import argparse
import multiprocessing as mp


def main(args):
    dm = EventDM(data_dir='dataset', data_name='ncaltech101', batch_size=4)
    dm.prepare_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--data_name', type=str, default='ncaltech101')
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    main(args)