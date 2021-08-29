import os
import numpy as np
import time
import signal

import torch
from torch.utils.data import DataLoader

from dataload.kitti_dataloader import KittiDataset
from trainer_Kitti import ModelTrainer
from single_stage_detector import SingleStageDetector


from pathlib import Path
from attrdict import AttrDict
from config import KITTICONFIG
from utils.Visualize import Log
from tensorboardX import SummaryWriter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True



class KittiConfig(KITTICONFIG):
    mode = "loading"
    is_training = True

    # Choose here if you want to start training from a previous cfg.snapshot (None for new training)
    # snapshot = 'Log_2021-01-12_05-39-25'   #cfg.snapshot = 'Log_2020-03-19_19-53-27'
    snapshot = ''

    # Path of the result folder
    if not snapshot:
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)
    else:
        saving_path = os.path.join("results", snapshot)

    checkpoint_dir = os.path.join(saving_path, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = os.path.join(checkpoint_dir, "model.pkl")

    # Visulize
    log = Log(saving_path)
    writer = SummaryWriter(os.path.join(saving_path, 'tensorboardx'))


if __name__ == '__main__':

    cfg = KittiConfig()
    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    ##############
    # Prepare Data
    ##############

    cfg.log.out('Data Preparation')
    cfg.log.out('****************')

    # Initialize datasets
    train_dataset = KittiDataset(cfg, cfg.mode, split = 'training', img_list = 'train', is_training = cfg.is_training)
    val_dataset = KittiDataset(cfg, cfg.mode, split = 'training', img_list = 'val', is_training = False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn = train_dataset.load_batch,
        drop_last = False, sampler = None, timeout = 0
    )

    val_loader = DataLoader(
        val_dataset,
        #batch_size=cfg.batch_size,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.val_batch_size,
        pin_memory=True,
        collate_fn=val_dataset.load_batch,
        drop_last=False, sampler=None, timeout=0
    )

    ###############
    # Snapshot parameters
    ###############
    cfg.log.out('Snapshot Preparation')
    cfg.log.out('****************')

    if cfg.snapshot:
        # Find related snapshot in the chosen training folder
        cfg.snapshot = os.path.join('results', cfg.snapshot, 'checkpoints.pkl')

    ###############
    # Architectures network
    ###############

    cfg.log.out('Model Preparation')
    cfg.log.out('*****************')

    model = SingleStageDetector(cfg, is_training=True)

    trainer = ModelTrainer()

    cfg.log.out('Start training')
    cfg.log.out('**************')

    
    ###############
    # Training
    ###############

    try:
        trainer.train(model, train_loader, val_loader, cfg)
    except:
        cfg.log.out('Caught an error')
        os.kill(os.getpid(), signal.SIGINT)


    cfg.log.out('Forcing exit now')
    cfg.log.close()
    os.kill(os.getpid(), signal.SIGINT)


