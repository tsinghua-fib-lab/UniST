import argparse

import random

import os

from model_prompt import mae_vit_ST

from train import TrainLoop

import setproctitle

import torch

from DataLoader_ST import  data_load_main
from utils import *


import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import resource


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
        lr = 5e-5,
        task = 'short',
        few_ratio = 1.0,
        weight_decay=0.0,
        batch_size=2,
        early_stop = 10,
        early_stop1 = 3,
        early_stop2 = 10,
        log_interval=10,
        save_interval=50,
        total_epoches = 1000,
        device_id='0',
        machine = 'LM1',
        lr_anneal_steps = 1,
        patch_size = 2,
        random=True,
        t_patch_size = 2,
        size = 'middle',
        clip_grad = 0.02,
        mask_strategy = 'causal', # ['ransom','causal','frame','tube']
        mask_strategy_random = 'none',
        mode='finetuning',
        file_load_path = '',
        min_lr = 5e-6,
        dataset = 'Crowd',
        his_len = 4,
        pred_len = 8,
        lr_anneal_steps1 = 100,
        total_epoches1 = 150,
        lr_anneal_steps2 = 200,
        total_epoches2 = 300,
        lr1 = 1e-3,
        lr2 = 5e-5,
        batch_size_taxibj = 64,
        batch_size_nj = 128,
        batch_size_nyc = 256,
        pos_emb = 'SinCos',#['trivial','SinCos']
        no_qkv_bias = 0,
        used_data = 'itself',
        finetune_part = 1, 
        prompt_ST = 1,
        num_memory_spatial = 32,
        num_memory_temporal = 32,
        conv_num = 3,
        prompt_content = 's_p_c',
        stage = 0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()

    setproctitle.setproctitle('prompt_tuning')

    setup_init(100)
    
    if args.few_ratio < 1 and args.few_ratio > 0:
        args.mode='few-shot'

    if args.few_ratio == 0.0:
        args.mode='zero-shot'

    args.folder = 'Prompt_Mode_{}_Dataset_{}_His_{}_Pred_{}'.format(args.mode, args.dataset, args.his_len, args.pred_len)
    
    args.mask_ratio = args.pred_len / (args.pred_len+args.his_len) - 1e-4

    args.model_path = './experiments/{}'.format(args.folder)
    logdir = "./logs/{}/".format(args.folder)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    with open(args.model_path+'result_all.txt', 'w') as f:
        f.write('start training\n')

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    data, test_data, val_data, args.scaler = data_load_main(args)

    model = mae_vit_ST(args=args).to(device)
   
    model.load_state_dict(torch.load('./experiments/{}/model_save/model_best.pkl'.format(args.file_load_path),map_location=device), strict=False)
    print('pretrained model loaded')

    model = model.to(device)

    # the first stage, only update newly initialized network parameters
    for k, param in model.named_parameters():
        if 'st_prompt' not in k and 'spatial_patch' not in k:
            param.requires_grad = False

    args.lr = args.lr1      
    args.min_lr = args.lr2
    args.lr_anneal_steps = args.lr_anneal_steps1
    args.total_epoches = args.total_epoches1
    args.stage = 1
    args.early_stop = args.early_stop1

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        data=data,
        val_data = val_data,
        test_data=test_data, 
        device=device
    ).run_loop()

    # the second stage, finetune networks apart from attn and mlp
    for k, param in model.named_parameters():
        if args.finetune_part == 1:
            if 'norm' in k or 'decoder_pred' in k or 'pos_emb' in k or 'Embedding' in k:
                param.requires_grad = True

        elif args.finetune_part == 2:
            param.requires_grad = True # all finetuned


    args.lr = args.lr2
    args.min_lr = args.lr2 * 0.1
    args.lr_anneal_steps = args.lr_anneal_steps2
    args.total_epoches = args.total_epoches2
    args.stage = 2
    args.early_stop = args.early_stop2

    TrainLoop(
        args = args,    
        writer = writer,
        model=model,
        data=data,
        val_data = val_data,
        test_data=test_data, 
        device=device
    ).run_loop()

if __name__ == "__main__":
    main()
