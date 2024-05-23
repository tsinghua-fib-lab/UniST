import argparse

import random

import os

from model import mae_vit_ST

from train import TrainLoop

import setproctitle

import torch

from DataLoader import data_load_main
from utils import *


import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")


def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-3,
        task = 'short',
        early_stop = 20,
        weight_decay=1e-6,
        batch_size=256,
        log_interval=20,
        total_epoches = 1000,
        device_id='0',
        machine = 'machine_name',
        mask_ratio = 0.5,
        lr_anneal_steps = 500,
        patch_size = 2,
        random=True,
        t_patch_size = 2,
        size = 'middle',
        clip_grad = 0.05,
        mask_strategy = 'random', # ['random','causal','frame','tube']
        mask_strategy_random = 'batch', # ['none','batch']
        mode='training',
        file_load_path = '',
        min_lr = 1e-5,
        dataset = 'Crowd',
        stage = 0,
        no_qkv_bias = 0,
        batch_size_taxibj = 128,
        batch_size_nj = 64,
        batch_size_nyc = 256,
        pos_emb = 'SinCos',
        used_data = '',
        process_name = 'process_name',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)

    data, test_data, val_data, args.scaler = data_load_main(args)

    args.folder = 'Dataset_{}/'.format(args.dataset)

    args.model_path = './experiments/{}'.format(args.folder) 
    logdir = "./logs/{}".format(args.folder)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    print('start data load')

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    model = mae_vit_ST(args=args).to(device)

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        data=data,
        test_data=test_data, 
        val_data=val_data,
        device=device
    ).run_loop()


if __name__ == "__main__":
    main()