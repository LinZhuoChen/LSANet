### LSANet: Feature Learning on Point Sets by Local Spatial Aware Layer
The official implementation of [LSANet](<https://arxiv.org/abs/1905.05442>) in tensorflow.

Thanks to [Xuan-Yi Li](https://github.com/meteorshowers),and [Deng-Ping Fan](https://github.com/DengPingFan)'s help. 

#### Introduction

We propose a new network layer, named **Local Spatial Aware** (LSA) Layer, to model geometric structure
in local region accurately and robustly. Each feature extracting operation in LSA layer is related to **Spatial Distribution Weights** (SDWs), which are learned based on the spatial distribution in local region, to establish a strong link with inherent geometric shape. The experiments show that our LSA-based network, named **LSANet**, can achieve on par or better performance than the state-of-the-art methods when evaluating on the challenging benchmark datasets. The network architecture of LSANet and LSA module are shown below.

![LSANet](/figure/LSANet.png)

![LSA module](/figure/LSA_module.png)

#### Installation

The code is based on  [PointNet++](https://github.com/charlesq34/pointnet2). Please follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators.

#### Usage

------

##### Classification

ModelNet40 dataset can be downloaded [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).

To train a LSANet for classification, please run

```bash
python train_multi_gpu.py
```

##### Part segmentation

ShapeNet dataset can be downloaded [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). To train a LSANet for segmentation, please run

```
cd part_seg
python train_multi_gpu_one_hot.py
```

#### Useful links

[Awesome point set learning](https://github.com/LinZhuoChen/awesome-point-cloud-learning)

#### Citation

Please cite this paper if it is helpful to your research,

```
@article{chen2019lsanet,
  title={LSANet: Feature Learning on Point Sets by Local Spatial Aware Layer},
  author={Chen, Lin-Zhuo and Li, Xuan-Yi and Fan, Deng-Ping and Cheng, Ming-Ming and Wang, Kai and Lu, Shao-Ping},
  journal={arXiv preprint arXiv:1905.05442},
  year={2019}
}
```

