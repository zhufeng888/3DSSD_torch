# Pytorch implementation of 3DSSD(2020CVPR)



Original Paper:
[3DSSD: Point-based 3D Single Stage Object Detector](https://arxiv.org/abs/2002.10187)

Original Code:
https://github.com/dvlab-research/3DSSD

The data and our trained model can download from the baiduyunlink:

链接: https://pan.baidu.com/s/1j6U6QOIMHpOPeYAL3QG2tQ  密码: fs7j
 
 
 The process we try to make it easier to read and the core code like network model we keep consistent with original code .
  

## 3DSSD_torch Usage

### 1. Installation

Ubuntu18.04 

cuda10.1

cudnn7.6.5

pytorch1.7.1 

torchvision0.8.2

tensorflow-gpu2.3.0

tensorboard2.4.1

attrdict

tensorflow is necessary for evaluation. and make pointnet++ follow as:
	
	cd pointnet2
	python setup.py install
	

<br/>

### 2. Dataset

Please refer to [Original Code ](https://github.com/dvlab-research/3DSSD) or our baiduyunlink.



```
3DSSD_torch  #根目录
│   README.md
│
└───data    #存放数据集
│   │   KITTI
│   └──── train
│   	   │('Car',)  #存放车辆训练集
│   └──── val
│   	   │('Car',)  #存放车辆验证集
│
│
└───dataset    #额外的数据集
│   │   object
│   └──── training
│          │label_2
│          │planes

│ 
└───result   #存放训练信息日志和模型
│
└───model  
│   │   checkpoint    
│   └──── model.pkl	#训练好的模型
│
└───dataload  
│   │   kitti_dataloader.py    #数据加载
│
└───pointnet2  
│   │   setup.py    #编译pointnet++
│
└───utils
│    │tf_ops   #评估代码
│
└───builder   
│  
│   config.py	#训练和验证过程公共的配置文件
│   my_train_Kitti.py	#配置和启动训练
│   trainer_Kitti.py	
│   my_val_Kitti.py  #配置和启动验证
│   valer_Kitti.py  
│   single_stage_detector.py  #网络模型



```

<br/>

### 3. Train


run

```bash
python my_train_Kitti.py
 
```

Parameters like batch_size,  lr, etc  could be changed in config.py.


<br/>

### 4. ValDataset Evaluation

run

```bash
python my_val_Kitti.py
 
```

<br/>



### 5. Result




Result from official:

|  Methods   | Easy  AP |Moderate AP  |Hard AP  |
|  ----  | ----  | ----  | ----  
| 3DSSD  | 91.71 |83.30 |80.44 |
| PointRCNN  | 88.91 |79.88 |78.37 |

Result we trained with official 3DSSD code:
```bash
precision_image:
[[ 0.96612436 0.93489265 0.92814064]
[-1. -1. -1. ]
[-1. -1. -1. ]]
precision_ground:
[[ 0.9327345 0.89477235 0.8677206 ]
[-1. -1. -1. ]
[-1. -1. -1. ]]
precision_3d:
[[ 0.91790676 0.83215886 0.80374014]
[-1. -1. -1. ]
[-1. -1. -1. ]]
```

Result we trained with 3DSSD_torch:
```bash
precision_image:
[[ 0.9635466   0.93081504  0.9241562 ]
 [-1.         -1.         -1.        ]
 [-1.         -1.         -1.        ]]
precision_ground:
[[ 0.928524    0.8910494   0.86533344]
 [-1.         -1.         -1.        ]
 [-1.         -1.         -1.        ]]
precision_3d:
[[ 0.9089438  0.8225948  0.7959043]
 [-1.        -1.        -1.       ]
 [-1.        -1.        -1.       ]]
```
 The ap difference is within 1%. Maybe a difference from framework or training.


<br/>

## Acknowledgement

Thanks for [Official](https://github.com/dvlab-research/3DSSD) providing original paper and code.

Thanks for [3DSSD-pytorch-openPCDet](https://github.com/qiqihaer/3DSSD-pytorch-openPCDet)
providing an implementation of 3DSSD in Pytorch and  encapsulating it in [openPCDet](https://github.com/open-mmlab/OpenPCDet). The pytorch F-FPS we get is  from it.


  <br/>

  <br/>

