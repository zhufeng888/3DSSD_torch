import torch
import time
import os
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.autograd import Variable
import tqdm
import pickle
import dataload.maps_dict as maps_dict
from utils import box_3d_utils
from utils.anchors_util import project_to_image_space_corners

from utils.tf_ops.evaluation.tf_evaluate import evaluate, calc_iou
import tensorflow as tf


class ModelTrainer():
    def train(self, cfg, model, train_loader, val_loader, val_dataset=None):

        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        start_epoch = 0
        if cfg.snapshot:
            checkpoint = torch.load(cfg.checkpoint)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            cfg.log.out('Load model successfully: %s' % (cfg.checkpoint))

        metric = {
            'gt_num': 0,
        }
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        cfg.log.out('*************** Start Evaluation *****************')

        model.cuda()
        model.eval()

        progress_bar = tqdm.tqdm(total=len(val_loader), leave=True, desc='eval', dynamic_ncols=True)

        obj_detection_list = []
        obj_detection_num = []
        obj_detection_name = []

        cur_model_path = cfg.saving_path
        self.log_dir = cfg.saving_path
        self.label_dir = os.path.join(cfg.ROOT_DIR,cfg.DATASET.KITTI.BASE_DIR_PATH, "training", 'label_2')

        # evaluation tools
        self.last_eval_model_path = None
        self.last_best_model = None
        self.last_best_result = -1
        # Start validation loop
        for batch in tqdm.tqdm(val_loader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            # Forward pass
            with torch.no_grad():
                layer_out = model(cfg, batch, bn_decay=None)

            pred_bbox_3d_op = layer_out[maps_dict.PRED_3D_BBOX][-1].squeeze(0).cpu().numpy()
            pred_cls_score_op = layer_out[maps_dict.PRED_3D_SCORE][-1].squeeze(0).cpu().numpy()
            pred_cls_category_op = layer_out[maps_dict.PRED_3D_CLS_CATEGORY][-1].squeeze(0).cpu().numpy() - 1

            calib_P, sample_name = [batch['frame_calib_p2'],batch['sample_name']]

            sample_name = int(sample_name[0])
            calib_P = calib_P[0].cpu()
            select_idx = np.where(pred_cls_score_op >= cfg.cls_thresh)[0]
            select_idx = select_idx.tolist()
            pred_cls_score_op = pred_cls_score_op[select_idx]
            pred_cls_category_op = pred_cls_category_op[select_idx]
            pred_bbox_3d_op = pred_bbox_3d_op[select_idx]
            pred_bbox_corners_op = box_3d_utils.get_box3d_corners_helper_np(
                pred_bbox_3d_op[:, :3], pred_bbox_3d_op[:, -1], pred_bbox_3d_op[:, 3:-1])
            pred_bbox_2d = project_to_image_space_corners(
                pred_bbox_corners_op, calib_P)

            obj_num = len(pred_bbox_3d_op)
            obj_detection = np.zeros([obj_num, 14], np.float32)

            if 'Car' not in cfg.DATASET.KITTI.CLS_LIST:
                pred_cls_category_op += 1
            obj_detection[:, 0] = pred_cls_category_op
            obj_detection[:, 1:5] = pred_bbox_2d
            obj_detection[:, 6:9] = pred_bbox_3d_op[:, :3]
            obj_detection[:, 9] = pred_bbox_3d_op[:, 4]  # h
            obj_detection[:, 10] = pred_bbox_3d_op[:, 5]  # w
            obj_detection[:, 11] = pred_bbox_3d_op[:, 3]  # l
            obj_detection[:, 12] = pred_bbox_3d_op[:, 6]  # ry
            obj_detection[:, 13] = pred_cls_score_op

            obj_detection_list.append(obj_detection)
            obj_detection_name.append(os.path.join(
                self.label_dir, '%06d.txt' % sample_name))
            obj_detection_num.append(obj_num)

        obj_detection_list = np.concatenate(obj_detection_list, axis=0)
        obj_detection_name = np.array(obj_detection_name, dtype=np.string_)
        obj_detection_num = np.array(obj_detection_num, dtype=np.int32)


        precision_img, aos_img, precision_ground, aos_ground, precision_3d, aos_3d = evaluate(
            obj_detection_list, obj_detection_name, obj_detection_num)


        result_list = [precision_img, aos_img,
                       precision_ground, aos_ground, precision_3d, aos_3d]

        cur_result = self.logger_and_select_best_map(
            result_list, cfg)

        if cur_result > self.last_best_result:
            # if cur_result is larger, save the current weight model
            if self.last_best_model is not None:
                # if last model is not none, then we remove that
                last_best_model_name = os.path.basename(
                    self.last_best_model)
                last_best_model_path = os.path.join(
                    self.log_dir, last_best_model_name)
                os.system('rm \"%s\".*' % os.path.join(last_best_model_path))

            #os.system('cp \"%s\".* \"%s\"' %
            #          (os.path.join(cur_model_path), os.path.join(self.log_dir)))
            self.last_best_model = cur_model_path
            self.last_best_result = cur_result

            cfg.log.out(self.last_eval_model_path)

        progress_bar.close()
        cfg.log.out('****************Evaluation done.*****************')

    def logger_and_select_best_map(self, result_list, cfg):
        """
            cfg: a function to print final result
        """
        precision_img_op, aos_img_op, precision_ground_op, aos_ground_op, precision_3d_op, aos_3d_op = result_list

        cfg.log.out('precision_image:')
        # [NUM_CLASS, E/M/H], NUM_CLASS: Car, Pedestrian, Cyclist
        #precision_img_res = precision_img_op[:, :, 1:]
        precision_img_res = precision_img_op.numpy()[:, :, 1:]
        precision_img_res = np.sum(precision_img_res, axis=-1) / 40.
        cfg.log.out(str(precision_img_res))

        cfg.log.out('precision_ground:')
        #precision_ground_res = precision_ground_op[:, :, 1:]
        precision_ground_res = precision_ground_op.numpy()[:, :, 1:]
        precision_ground_res = np.sum(precision_ground_res, axis=-1) / 40.
        cfg.log.out(str(precision_ground_res))

        cfg.log.out('precision_3d:')
        precision_3d_res = precision_3d_op.numpy()[:, :, 1:]
        precision_3d_res = np.sum(precision_3d_res, axis=-1) / 40.
        cfg.log.out(str(precision_3d_res))

        if 'Car' in cfg.DATASET.KITTI.CLS_LIST:
            cur_result = precision_3d_res[0, 1]
        else:  # Pedestrian and Cyclist
            cur_result = (precision_3d_res[1, 1] + precision_3d_res[2, 1]) / 2.

        return cur_result












