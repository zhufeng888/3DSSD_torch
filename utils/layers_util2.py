#import tensorflow as tf
import numpy as np
# import utils.tf_util as tf_util
# import utils.model_util as model_util

import torch.nn as nn
import torch
import torch.nn.functional as F
import pointnet2.pointnet2_utils as pointnet2_utils
import pointnet2.pytorch_utils as pt_utils
from utils.model_util import calc_square_dist, nn_distance

# from utils.tf_ops.grouping.tf_grouping import *
# from utils.tf_ops.sampling.tf_sampling import *
# from utils.tf_ops.interpolation.tf_interpolate import *


class Vote_layer(nn.Module):
    def __init__(self, cfg, mlp_list, bn, is_training, pre_channel):
        super(Vote_layer, self).__init__()
        self.cfg = cfg
        self.mlp_list = list(mlp_list)
        self.bn = bn
        self.is_training = is_training
        self.layer_list = nn.ModuleList()

        self.mlp_list = [pre_channel] + list(self.mlp_list)
        for i in range(len(self.mlp_list) - 1 ):
            self.layer_list.append(pt_utils.Conv1d(self.mlp_list[i], self.mlp_list[i+1], bn=False))
            self.layer_list.append(nn.BatchNorm1d(self.mlp_list[i+1],eps=1e-3, affine=False))

        self.ctr_reg = pt_utils.Conv1d(self.mlp_list[-1], 3, activation=None, bn=False)
        self.min_offset = torch.tensor(self.cfg.MODEL.MAX_TRANSLATE_RANGE).float().view(1, 1, 3)


    def forward(self, xyz, points, bn_decay):

        points = points.transpose(1, 2)

        for i in range(len(self.mlp_list)):
            if bn_decay is not None:
                self.layer_list[i].momentum = (1 - bn_decay)
            points = self.layer_list[i](points)
        ctr_offsets = self.ctr_reg(points)

        ctr_offsets = ctr_offsets.transpose(1, 2)
        points = points.transpose(1, 2)

        min_offset = torch.tensor(self.cfg.MODEL.MAX_TRANSLATE_RANGE).float().view(1, 1, 3).repeat((points.shape[0], points.shape[1], 1)).to(points.device)

        limited_ctr_offsets = torch.where(ctr_offsets > min_offset, ctr_offsets, min_offset)
        min_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset, limited_ctr_offsets, min_offset)
        xyz = xyz + limited_ctr_offsets

        return xyz, points, ctr_offsets

class Pointnet_sa_module_msg(nn.Module):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int -- points sampled in farthest point sampling
            radius_list: list of float32 -- search radius in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp_list: list of list of int32 -- output size for MLP on each point
            fps_method: 'F-FPS', 'D-FPS', 'FS'
            fps_start_idx:
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''

    def __init__(self, cfg, radius_list, nsample_list,
                 mlp_list, is_training, bn_decay, bn,
                 fps_sample_range_list, fps_method_list, npoint_list, use_attention, scope,
                 dilated_group, aggregation_channel=None, pre_channel=0,
                 debugging=False,
                 eps=1e-5):
        super().__init__()
        self.cfg = cfg
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_list = list(mlp_list)
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.fps_sample_range_list = fps_sample_range_list
        self.fps_method_list = fps_method_list
        self.npoint_list = npoint_list
        self.use_attention = use_attention
        self.scope = scope
        self.dilated_group = dilated_group
        self.aggregation_channel = aggregation_channel
        self.pre_channel = pre_channel

        self.layer_list = nn.ModuleList()

        for i in range(len(self.radius_list)):
            self.mlp_list[i] = [self.pre_channel + 3] + list(self.mlp_list[i])
            tmp_layer_list = nn.ModuleList()
            for j in range(len( self.mlp_list[i]) -1 ):
                if False:
                    tmp_layer_list.append(pt_utils.Conv1d(self.mlp_list[i][j], self.mlp_list[i][j+1], bn=False))
                    tmp_layer_list.append(nn.BatchNorm1d(self.mlp_list[i][j+1], eps=1e-3, affine=False))
                else:
                    tmp_layer_list.append(pt_utils.Conv2d(self.mlp_list[i][j], self.mlp_list[i][j + 1], bn=False))
                    tmp_layer_list.append(nn.BatchNorm2d(self.mlp_list[i][j + 1], eps=1e-3, affine=False))

            self.layer_list.append(tmp_layer_list)


        self.AGGREGATION_SA_FEATURE = True
        if self.AGGREGATION_SA_FEATURE and (len(self.mlp_list) != 0):
            input_channel = 0
            for mlp_tmp in self.mlp_list:
                input_channel += mlp_tmp[-1]
            self.aggregation_layer = pt_utils.Conv1d(input_channel, aggregation_channel, bn=False)
            self.aggregation_layer_bn = nn.BatchNorm1d(aggregation_channel,eps=1e-3, affine=False)
            

    def forward(self, xyz, points, former_fps_idx, vote_ctr, bn_decay):

        bs = xyz.shape[0]
        num_points = xyz.shape[1]

        cur_fps_idx_list = []
        last_fps_end_index = 0
        for fps_sample_range, fps_method, npoint in zip(self.fps_sample_range_list, self.fps_method_list, self.npoint_list):
            if fps_sample_range < 0:
                fps_sample_range_tmp = fps_sample_range + num_points + 1
            else:
                fps_sample_range_tmp = fps_sample_range
            tmp_xyz = xyz[:, last_fps_end_index:fps_sample_range_tmp, :].contiguous()
            tmp_points = points[:, last_fps_end_index:fps_sample_range_tmp, :].contiguous()
            if npoint == 0:
                last_fps_end_index += fps_sample_range
                continue
            if vote_ctr is not None:
                npoint = vote_ctr.shape[1]
                fps_idx = torch.arange(npoint).int().view(1, npoint).repeat((bs, 1)).to(tmp_xyz.device)
            elif fps_method == 'FS':
                features_for_fps = torch.cat([tmp_xyz, tmp_points], dim=-1)
                # dist1 = nn_distance(tmp_xyz, tmp_xyz)
                # dist2 = calc_square_dist(tmp_xyz, tmp_xyz, norm=False)
                features_for_fps_distance = calc_square_dist(features_for_fps, features_for_fps, norm=False)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                fps_idx_2 = pointnet2_utils.furthest_point_sample(tmp_xyz, npoint)
                fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)  # [bs, npoint * 2]
            elif npoint == tmp_xyz.shape[1]:
                fps_idx = torch.arange(npoint).int().view(1, npoint).repeat((bs, 1)).to(tmp_xyz.device)
            elif fps_method == 'F-FPS':
                features_for_fps = torch.cat([tmp_xyz, tmp_points], dim=-1)
                features_for_fps_distance = calc_square_dist(features_for_fps, features_for_fps, norm=False)
                features_for_fps_distance = features_for_fps_distance.contiguous()
                fps_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
            else: # D-FPS
                fps_idx = pointnet2_utils.furthest_point_sample(tmp_xyz, npoint)

            fps_idx = fps_idx + last_fps_end_index
            cur_fps_idx_list.append(fps_idx)
            last_fps_end_index += fps_sample_range
        fps_idx = torch.cat(cur_fps_idx_list, dim=-1)

        if former_fps_idx is not None:
            fps_idx = torch.cat([fps_idx, former_fps_idx], dim=-1)

        if vote_ctr is not None:
            vote_ctr_transpose = vote_ctr.transpose(1, 2).contiguous()
            new_xyz = pointnet2_utils.gather_operation(vote_ctr_transpose, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        new_points_list = []
        points = points.transpose(1, 2).contiguous()
        xyz = xyz.contiguous()
        for i in range(len(self.radius_list)):
            nsample = self.nsample_list[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.0
                else:
                    min_radius = self.radius_list[i-1]
                max_radius = self.radius_list[i]
                idx = pointnet2_utils.ball_query_dilated(max_radius, min_radius, nsample, xyz, new_xyz)
            else:
                radius = self.radius_list[i]
                idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)

            xyz_trans = xyz.transpose(1, 2).contiguous()
            grouped_xyz = pointnet2_utils.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
            grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

            #features=torch.cat([xyz_trans, points], dim=1)
            features = points

            #grouped_points = pointnet2_utils.grouping_operation(features, idx)


            #if grouped_points.shape[1] <= 64:
                #grouped_points = grouped_points - grouped_points[:,:,:,0].unsqueeze(-1)



            if points is not None:
                grouped_points = pointnet2_utils.grouping_operation(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                grouped_points = grouped_xyz
            #grouped_points= torch.Size([4, 4, 4096, 32])

            for j in range(len(self.layer_list[i])):
                if hasattr(self.layer_list[i][j], 'momentum') and bn_decay is not None:
                    self.layer_list[i][j].momentum = (1-bn_decay)
                grouped_points = self.layer_list[i][j](grouped_points)

            new_points = F.max_pool2d(grouped_points, kernel_size=[1, grouped_points.size(3)])
            new_points_list.append(new_points.squeeze(-1))

        if len(new_points_list) > 0:
            new_points_concat = torch.cat(new_points_list, dim=1)
            if self.AGGREGATION_SA_FEATURE:
                new_points_concat = self.aggregation_layer(new_points_concat)
                if bn_decay is not None:
                    self.aggregation_layer_bn.momentum = (1 - bn_decay)
                new_points_concat = self.aggregation_layer_bn(new_points_concat)

        else:
            new_points_concat = pointnet2_utils.gather_operation(points, fps_idx)

        new_points_concat = new_points_concat.transpose(1, 2).contiguous()

        return new_xyz, new_points_concat, fps_idx

