import argparse
import random
import os
from model import UniST_model
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
        # experimental settings
        task = 'short',
        dataset = 'Crowd',
        mode='training', # ['training','prompting','testing']
        file_load_path = '',
        used_data = '',
        process_name = 'process_name',
        prompt_ST = 0,
        his_len = 6,
        pred_len = 6,
        few_ratio = 0.5,
        stage = 0,

        # model settings
        mask_ratio = 0.5,
        patch_size = 2,
        t_patch_size = 2,
        size = 'middle',
        no_qkv_bias = 0,
        pos_emb = 'SinCos',
        num_memory_spatial = 512,
        num_memory_temporal = 512,
        conv_num = 3,
        prompt_content = 's_p_c',

        # pretrain settings
        random=True,
        mask_strategy = 'random', # ['random','causal','frame','tube']
        mask_strategy_random = 'batch', # ['none','batch']
        
        # training parameters
        lr=1e-3,
        min_lr = 1e-5,
        early_stop = 5,
        weight_decay=1e-6,
        batch_size=256,
        log_interval=20,
        total_epoches = 200,
        device_id='0',
        machine = 'machine_name',
        clip_grad = 0.05,
        lr_anneal_steps = 200,
        batch_size_1 = 64,
        batch_size_2 = 32,
        batch_size_3 = 16,
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
    assert args.his_len + args.pred_len == args.seq_len

    args.folder = 'Dataset_{}_Task_{}_FewRatio_{}/'.format(args.dataset, args.task, args.few_ratio)

    if args.mode in ['training','prompting']:
        if args.prompt_ST != 0:
            args.folder = 'Prompt_'+args.folder
        else:
            args.folder = 'Pretrain_'+args.folder
    else:
        args.folder = 'Test_'+args.folder

    args.model_path = './experiments/{}'.format(args.folder) 
    logdir = "./logs/{}".format(args.folder)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)
    device = dev(args.device_id)

    model = UniST_model(args=args).to(device)

    if args.prompt_ST==1:
        if args.file_load_path != '':
            model.load_state_dict(torch.load('./experiments/{}/model_save/model_best.pkl'.format(args.file_load_path),map_location=device), strict=False)
            print('pretrained model loaded') 
        args.mask_strategy_random = 'none'
        args.mask_strategy = 'temporal'
        args.mask_ratio = (args.pred_len+0.0) / (args.pred_len+args.his_len)

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        data=data,
        test_data=test_data, 
        val_data=val_data,
        device=device,
        early_stop = args.early_stop,
    ).run_loop()

if __name__ == "__main__":
    main()