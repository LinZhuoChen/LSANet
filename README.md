### LSANet: Feature Learning on Point Sets by Local Spatial Attention
Thanks to [Xuan-Yi Li](https://github.com/meteorshowers),and [Deng-Ping Fan](https://github.com/DengPingFan)'s help. 
#### Introduction

We design a novel **Local Spatial Attention** (LSA) module to adaptively generate attention maps according to the spatial distribution of local regions. The feature learning process which integrates with these attention maps can effectively capture the local geometric structure. The experiments show that our LSA-based network, named **LSANet**, can achieve on par or better performance than the state-of-the-art methods when evaluating on the challenging benchmark datasets. The network architecture of LSANet and LSA module are shown below.

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

[Awesome point set learning] (https://github.com/LinZhuoChen/awesome-point-cloud-learning)