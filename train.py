import argparse
import os
import json
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.light_former import LightFormerPredictor

# import sys
# sys.path.append('.')

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='train intention network.')
    parser.add_argument('-cfg', '--config', type=str, default='', required=True, help='config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('-save', '--save_path', type=str, default='./result', help='path to save model')
    parser.add_argument('-log', '--log_dir', type=str, default='./log', help='log directory')
    parser.add_argument('-d', '--devices', type=int, default=1, help='0: cpu, n: n num of devices, -1: all devices')
    parser.add_argument('-n', '--node', type=int, default=1, help='num of nodes across multi machines')
    parser.add_argument('--resume_weight_only',
                        dest='resume_weight_only',
                        action='store_true',
                        default=False, # False
                        help='resume only weights from chekpoint file')
    parser.add_argument('-v', '--verbose', type=bool, default=False, required=False, help='print more console statements for debugging')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # must set seed everything in multi node multi gpus training
    seed: int = 42
    seed_everything(seed)

    # parse command line arguments
    args: argparse.ArgumentParser = parse_args()
    print("\nargs:", args)

    # load config
    config = None
    config_file = args.config
    print(f'\nUsing config: {config_file}')
    with open(config_file, 'r') as f:
        config = json.load(f)
    if args.verbose:
        print("\nconfig:", config)

    # set device count
    devices = args.devices
    if devices == 0:
        print('Gpu not specified, exit normally')
        exit(0)

    # set node
    node_num = args.node
    if node_num < 1:
        print('Node num must be greater than 0')
        exit(0)

    # ModelCheckpoint callback saves best val_error_rate and last epoch
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=save_path,
                                          filename="checkpoint-{epoch:04d}-{val_loss:.5f}",
                                          save_weights_only=False,
                                          mode='min',
                                          save_top_k=10)

    # set checkpoint path
    checkpoint_file = args.checkpoint
    if checkpoint_file is not None:
        print(f'Using checkpoint: {checkpoint_file}')

    # create logger
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(log_dir, name=config['model_name'])

    # LearningRateMonitor automatically monitors and logs learning rate for lr schedulers.
    lr_monitor = LearningRateMonitor(logging_interval='step',
                                     log_momentum=True,
                                     log_weight_decay=True)

    # formatting
    print()

    # create Trainer
    trainer = pl.Trainer(
        accelerator=config['training']['accelerator'],
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=False,
        devices=devices,
        max_epochs=config['training']['epoch'],
        logger=tb_logger,
        gradient_clip_algorithm=config['optim']['gradient_clip_algorithm'],
        log_every_n_steps=config['log_every_n_steps'],
        # num_nodes=node_num,
        # strategy='ddp',
        # sync_batchnorm=True,
        # val_check_interval=config['validation']['check_interval'],
        # limit_val_batches=config['validation']['limit_batches'],
        # replace_sampler_ddp=True,
        # auto_lr_find=True
    )

    # run training
    resume_weight_only = args.resume_weight_only
    if resume_weight_only:
        predictor = LightFormerPredictor.load_from_checkpoint(config=config,
                                                              checkpoint_path=checkpoint_file,
                                                              strict=True)
        trainer.fit(predictor, ckpt_path=checkpoint_file)
    else:
        predictor = LightFormerPredictor(config=config)
        trainer.fit(predictor)
