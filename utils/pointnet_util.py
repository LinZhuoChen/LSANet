# coding:utf-8
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, bn, is_training, bn_decay, mlp, knn=False, use_xyz=True,
                     xyz_feature=None, end=False, use_edge_feature=False):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization

    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    if use_edge_feature == False:
        return new_xyz, new_points, idx, grouped_xyz
    # [batch_size, npoint, 1, F]
    if xyz_feature == None:
        xyz_feature = xyz

    xyz_feature = group_point(xyz_feature, idx)
    edge_feature = grouped_xyz
    for i, num_out_channel in enumerate(mlp):
        edge_feature = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                                      padding='VALID', stride=[1, 1],
                                      bn=bn, is_training=is_training,
                                      scope='xyz_feature_%d' % (i), bn_decay=bn_decay)
    output_feature = tf.concat([xyz_feature, edge_feature], axis=-1)
    if end == False:
        xyz_feature = tf_util.conv2d(output_feature, mlp[-1], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='xyz_feature2', bn_decay=bn_decay)
        # we can try sum and mean
        xyz_feature = tf.reduce_max(xyz_feature, axis=[2], keep_dims=True, name='maxpool')
        xyz_feature = tf.squeeze(xyz_feature, [2])
    return new_xyz, new_points, idx, output_feature, xyz_feature, grouped_xyz


def sample_and_group_all(xyz, points, bn, is_training, bn_decay, mlp, use_xyz=True, xyz_feature=None,
                         end=False, use_edge_feature=False):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz

    if use_edge_feature == False:
        return new_xyz, new_points, idx, grouped_xyz

    if xyz_feature == None:
        xyz_feature = xyz

    xyz_feature = tf.reshape(xyz_feature, (batch_size, 1, nsample, -1))
    edge_feature = grouped_xyz
    for i, num_out_channel in enumerate(mlp):
        edge_feature = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                                      padding='VALID', stride=[1, 1],
                                      bn=bn, is_training=is_training,
                                      scope='xyz_feature_%d' % (i), bn_decay=bn_decay)

    output_feature = tf.concat([xyz_feature, edge_feature], axis=-1)
    if end == False:
        xyz_feature = tf_util.conv2d(output_feature, mlp[-1], [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=bn, is_training=is_training,
                                     scope='xyz_feature5', bn_decay=bn_decay)

        xyz_feature = tf.reduce_max(xyz_feature, axis=[2], keep_dims=True, name='maxpool')
        xyz_feature = tf.squeeze(xyz_feature, [2])
    return new_xyz, new_points, idx, output_feature, xyz_feature, grouped_xyz

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)

        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1


def LSA_layer(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay,
                          scope, xyz_feature=None, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False,
                          end=False):
    ''' LSA layer
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            is_training: bool -- whether train this LSA layer
            bn_decay: float32 -- batch norm decay
            scope: scope in tensorflow
            xyz_feature: float32 -- feature from SFE
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, output_feature, xyz_feature, grouped_xyz = sample_and_group_all(xyz, points, bn,
                                                                                                      is_training,
                                                                                                      bn_decay, mlp2,
                                                                                                      use_xyz,
                                                                                                      xyz_feature, end,
                                                                                                      use_edge_feature=True)
        else:
            new_xyz, new_points, idx, output_feature, xyz_feature, grouped_xyz = sample_and_group(npoint, radius,
                                                                                                  nsample, xyz, points,
                                                                                                  bn, is_training,
                                                                                                  bn_decay, mlp2,
                                                                                                  knn, use_xyz,
                                                                                                  xyz_feature, end,
                                                                                                  use_edge_feature=True)
        # xyz Feature Embedding
        new_points = tf.concat([new_points, output_feature], axis=-1)

        channel = new_points.get_shape()[-1].value
        attention_xyz_1 = tf_util.conv2d(grouped_xyz, 64, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='xyz_attention_1', bn_decay=bn_decay,
                                         data_format=data_format)

        attention_xyz_2 = tf_util.conv2d(grouped_xyz, 64, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='xyz_attention_2', bn_decay=bn_decay,
                                         data_format=data_format)
        attention_xyz_2 = tf.reduce_mean(attention_xyz_2, axis=[2], keep_dims=True, name='meanpool')
        attention_xyz_2 = tf.tile(attention_xyz_2, [1, 1, nsample, 1])
        attention_xyz = tf.concat([attention_xyz_1, attention_xyz_2], axis=-1)

        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format)
            attention_xyz = tf_util.conv2d(attention_xyz, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='xyz_attention%d' % (i), bn_decay=bn_decay,
                                        data_format=data_format, activation_fn=tf.sigmoid)
            new_points = tf.multiply(new_points, attention_xyz)

        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool2')

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        # new_points = tf.concat([new_points, xyz_feature], axis=-1)
        return new_xyz, new_points, idx, xyz_feature

