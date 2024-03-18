from data.ncaltech101 import NCaltech101
from data.ncars import NCars
from models.lm import LNModel

from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
import argparse
import multiprocessing as mp
import torch.onnx

def main(args):
    # dm = NCaltech101(data_dir='dataset', data_name='ncaltech101', batch_size=args.batch_size)
    dm = NCars(data_dir='dataset', batch_size=args.batch_size)
    dm.setup()

    model = LNModel(lr=1e-3, weight_decay=5e-3, num_classes=dm.num_classes, batch_size=args.batch_size, input_dimension=dm.dim)
    wandb_logger = WandbLogger(project='event_classification', name='ncars', log_model='all')
    wandb_logger.watch(model, log='gradients')
    trainer = L.Trainer(max_epochs=100, log_every_n_steps=1, gradient_clip_val=0.0, accumulate_grad_batches=64, logger=wandb_logger)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/dataset')
    parser.add_argument('--data_name', type=str, default='gen1')
    parser.add_argument('--batch_size', type=int, default=1)

    mp.set_start_method('spawn', force=True)
    args = parser.parse_args()
    main(args)


