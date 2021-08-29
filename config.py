import torch
import os
import numpy as np
import time
import signal
from torch.utils.data import DataLoader


from utils.Visualize import Log
from tensorboardX import SummaryWriter

from attrdict import AttrDict


class KITTICONFIG():
    '''
    用于配置训练KITTICONFIG数据集的各项相关参数
    '''

    ####################
    # Dataset parameters
    ####################
    # If transform train dataset
    is_transform = False

    ####################
    # Environment parameters
    ####################
    gpu_nums = 1
    gpu_id = "0"

    #####################
    # Training parameters
    #####################

    # Number of batch
    batch_size = 4
    val_batch_size = 1

    # Number of CPU threads for the input pipeline
    num_workers = batch_size * gpu_nums

    # Maximal number (epochs and max_iter其中一个达到阈值就会停止训练)
    epochs = 120
    max_iter = 80700
    log_interval = 20
    check_interval = 3

    # Learning rate management
    learning_rate = 0.002
    STEPS = [64560]

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    POINT_CLOUD_RANGE = (-40, 40, -5, 3, 0, 70)
    POINT_FEATURE_ENCODING = AttrDict({
        "encoding_type": "absolute_coordinates_encoding",
        "used_feature_list": ['x', 'y', 'z', 'intensity'],
        "src_feature_list": ['x', 'y', 'z', 'intensity'], })

    DATA_AUGMENTOR = AttrDict({
        'DISABLE_AUG_LIST': ['placeholder'],
        'AUG_CONFIG_LIST': [
            AttrDict({'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'],
                      'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'],
                                  'filter_by_difficulty': [-1]},
                      'SAMPLE_GROUPS': ['Car:15', 'Pedestrian:10', 'Cyclist:10'], 'NUM_POINT_FEATURES': 4,
                      'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0],
                      'LIMIT_WHOLE_SCENE': False}),
            AttrDict({'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}),
            AttrDict({'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}),
            AttrDict({'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]})
        ]})

    DATA_PROCESSOR = [
        AttrDict({'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}),
        AttrDict({'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': AttrDict({'train': True, 'test': False})}),
    ]

    DATASET=AttrDict({
        'TYPE': 'KITTI',
        'POINT_CLOUD_RANGE': (-40, 40, -5, 3, 0, 70),
        'KITTI':AttrDict({
            'CLS_LIST': ('Car',),
            'BASE_DIR_PATH': 'dataset/KITTI/object',
            'TRAIN_LIST': 'dataset/KITTI/object/train.txt',
            'VAL_LIST': 'dataset/KITTI/object/val.txt',
            'SAVE_NUMPY_PATH': 'data/KITTI'
        })
    })


    TRAIN=AttrDict({
        'AUGMENTATIONS':AttrDict({
            'OPEN': True,
            'FLIP': True,
            'MIXUP':AttrDict({
                'OPEN': True,
                'SAVE_NUMPY_PATH': 'mixup_database/KITTI',
                'PC_LIST': 'train',
                'CLASS': ('Car',),
                'NUMBER': (15,)
            }),
            'EXPAND_DIMS_LENGTH': 0.1,
            'PROB_TYPE': 'Simultaneously',  # Simultaneously or Seperately
            'PROB': [0.5, 0.5, 0.5],
            'RANDOM_ROTATION_RANGE': 45 / 180 * np.pi,
            'RANDOM_SCALE_RANGE': 0.1,
            'SINGLE_AUG': AttrDict({
                'ROTATION_PERTURB': [-np.pi / 3, np.pi / 3],
                'CENTER_NOISE_STD': [1.0, 1.0, 0.],
                'RANDOM_SCALE_RANGE': [1.0, 1.0],
                'SCALE_3_DIMS': False,
                'FIX_LENGTH': False,
            })
        }),

        'CONFIG':AttrDict({
            'BATCH_SIZE': batch_size,
            'GPU_NUM': gpu_nums,
            'MAX_ITERATIONS': max_iter,
            'TOTAL_EPOCHS': epochs,
            'CHECKPOINT_INTERVAL': 807,
            'SUMMARY_INTERVAL': 10,
            'TRAIN_PARAM_PREFIX':[],
            'TRAIN_LOSS_PREFIX':[]
        }),
    })

    SOLVER=AttrDict({
        'TYPE': 'AdamW',
        'BASE_LR': 0.002,
        'LR_POLICY': 'steps_with_decay',
        'GAMMA': 0.1,
        'STEPS': [21600, 32400]
    })

    TEST=AttrDict({
        'WITH_GT': True,
        'TEST_MODE': 'mAP'
    })

    DATA_LOADER=AttrDict({
        'NUM_THREADS': 4  # GPU_NUM x BATCH_SIZE
    })

    MODEL = AttrDict({
        'POINTS_NUM_FOR_TRAINING': 16384,
        "ANGLE_CLS_NUM": 12,
        'MAX_TRANSLATE_RANGE': [-3.0, -2.0, -3.0],

        "POST_PROCESSING": AttrDict({
            'RECALL_THRESH_LIST': [0.3, 0.5, 0.7], 'SCORE_THRESH': 0.1,
            'OUTPUT_RAW_SCORE': False,
            'EVAL_METRIC': 'kitti',
            'NMS_CONFIG': {'MULTI_CLASSES_NMS': False, 'NMS_TYPE': 'nms_gpu',
                           'NMS_THRESH': 0.1,
                           'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500}}),
        # terget_assighner.py参数
        'FIRST_STAGE': AttrDict({
            'MAX_OUTPUT_NUM': 100,
            'NMS_THRESH': 0.1,

            'REGRESSION_METHOD': AttrDict({'TYPE': 'Dist-Anchor-free',
                                           'BIN_CLASS_NUM': 12}),
            'CLS_ACTIVATION': 'Sigmoid',
            'ASSIGN_METHOD': 'Mask',
            'CORNER_LOSS': True,
            'CLASSIFICATION_LOSS': AttrDict({
                'TYPE': 'Center-ness',
                'SOFTMAX_SAMPLE_RANGE': 10.0,
                "CENTER_NESS_LABEL_RANGE": [0.0, 1.0]}),
            'IOU_SAMPLE_TYPE': '3D',
            'MINIBATCH_NUM': -1,
            'MINIBATCH_RATIO': 0.25,
            'CLASSIFICATION_POS_IOU': 0.7,
            'CLASSIFICATION_NEG_IOU': 0.55,

            'PREDICT_ATTRIBUTE_AND_VELOCITY': False
        }),

        'NETWORK': AttrDict({
            'SYNC_BN': False,
            'USE_GN': False,
            'AGGREGATION_SA_FEATURE': True,
            'FIRST_STAGE':AttrDict({
                'ARCHITECTURE':[
                    [[0], [0], [0.2,0.4,0.8], [32,32,64], [[16,16,32], [16,16,32], [32,32,64]], True,
                     [-1], ['D-FPS'], [4096],
                     -1, False, 'SA_Layer', 'layer1', True, -1, 64], # layer1
                    [[1], [1], [0.4,0.8,1.6], [32,32,64], [[64,64,128], [64,64,128], [64,96,128]], True,
                     [-1], ['FS'], [512],
                     -1, False, 'SA_Layer', 'layer2', True, -1, 128], # layer2
                    [[2], [2], [1.6,3.2,4.8], [32,32,32], [[128,128,256], [128,192,256], [128,256,256]], True,
                     [512, -1], ['F-FPS', 'D-FPS'], [256, 256],
                     -1, False, 'SA_Layer', 'layer3', True, -1, 256], # layer3
                    # vote
                    [[3], [3], [], [], [], True,
                     [256, -1], ['F-FPS', 'D-FPS'], [256, 0],
                     -1, False, 'SA_Layer', 'vote', False, -1, 256],
                    [[4], [4], -1, -1, [128,], True,
                     [-1], [-1], [-1],
                     -1, -1, 'Vote_Layer', 'vote', False, -1, -1], # layer3-vote
                    # CG layer
                    [[3], [3], [4.8, 6.4], [16, 32], [[256,256,512], [256,512,1024]], True,
                     [-1], ['D-FPS'], [256],
                     -1, False, 'SA_Layer', 'layer4', False, 5, 512], # layer4
                  ],

                    'HEAD': [[[6], [6], 'conv1d', [128,], True, 'Det', '']]
            })
        }),
        'PATH' : AttrDict({
            'CHECKPOINT_DIR':'log',
            'EVALUATION_DIR': 'result'
        })
    })





