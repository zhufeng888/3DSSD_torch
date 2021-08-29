import torch
import os
import numpy as np
import time
import signal
from torch.utils.data import DataLoader

from valer_Kitti import ModelTrainer
from single_stage_detector import SingleStageDetector

from dataload.kitti_dataloader import KittiDataset
from attrdict import AttrDict
from config import KITTICONFIG
from utils.Visualize import Log
from tensorboardX import SummaryWriter

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True



class KittiConfig(KITTICONFIG):
    # If transform train dataset
    is_transform = False

    # Choose here if you want to start training from a previous cfg.snapshot (None for new training)
    # snapshot = 'Log_2021-01-12_05-39-25'   #cfg.snapshot = 'Log_2020-03-19_19-53-27'
    snapshot = 'model'
    saving_path = os.path.join("results", snapshot)
    checkpoint_dir = os.path.join(saving_path, "checkpoint")
    checkpoint = os.path.join(checkpoint_dir, "model.pkl")

    cls_thresh = 0.3

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

    val_dataset = KittiDataset(cfg, 'loading', split='training', img_list='val', is_training=False)

    val_loader = DataLoader(
        dataset=val_dataset,
        # batch_size=cfg.batch_size,
        batch_size=cfg.val_batch_size,
        num_workers=int(cfg.val_batch_size),
        pin_memory=True,
        collate_fn=val_dataset.load_batch,
        drop_last=False, sampler=None, timeout=0
    )

    cfg.log.out('Snapshot Preparation')
    cfg.log.out('****************')

    ###############
    # Architectures network
    ###############

    cfg.log.out('Model Preparation')
    cfg.log.out('*****************')

    model = SingleStageDetector(cfg, is_training=False)

    '''# 查看可优化的参数有哪些
    for name, param in model.named_parameters():
        if param.requires_grad:
            cfg.log.out(name +  " =")'''

    trainer = ModelTrainer()

    cfg.log.out('Start training')
    cfg.log.out('**************')

    ###############
    # Training
    ###############

    try:
        trainer.train(cfg, model, None, val_loader, val_dataset=val_dataset)
    except:
        cfg.log.out('Caught an error')
        os.kill(os.getpid(), signal.SIGINT)

    cfg.log.out('Forcing exit now')
    cfg.log.close()
    os.kill(os.getpid(), signal.SIGINT)




