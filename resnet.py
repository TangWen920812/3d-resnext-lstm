#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:35:28 2020

@author: tangwen
"""

import mxnet as mx
import numpy as np

def multi_focalloss(predict, label, gamma=2):
    label_onehot = mx.symbol.one_hot(label, depth=3)
    focal_loss = -(1 - predict * label_onehot)**gamma * mx.symbol.log10(predict + 1e-7)
    return mx.symbol.mean(focal_loss)

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.contrib.symbol.SyncBatchNorm(data=data, key=name + '_bn1', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1', use_global_stats=False)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1,1), stride=(1,1,1), pad=(0,0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.contrib.symbol.SyncBatchNorm(data=conv1, key=name + '_bn2', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2', use_global_stats=False)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3,3), stride=stride, pad=(1,1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.contrib.symbol.SyncBatchNorm(data=conv2, key=name + '_bn3', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3', use_global_stats=False)
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1,1), stride=(1,1,1), pad=(0,0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.contrib.symbol.SyncBatchNorm(data=data, key=name + '_bn1', fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1', use_global_stats=False)
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3,3), stride=stride, pad=(1,1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.contrib.symbol.SyncBatchNorm(data=conv1, key=name + '_bn2', fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2', use_global_stats=False)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet_3d(units, num_stage, filter_list, num_class, bottle_neck=True,
           bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    # data = mx.sym.Cast(data=data, dtype=np.float16)
    # label = mx.sym.Cast(data=label, dtype=np.float16)

    data = mx.contrib.symbol.SyncBatchNorm(data=data, key='bn_data', fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', use_global_stats=False)

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7,7,7), stride=(2,2,2), pad=(3,3,3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn0', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', use_global_stats=False)
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3,3,3), stride=(2,2,2), pad=(1,1,1), pool_type='max')

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn1', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', use_global_stats=False)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7,7,7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    flat = mx.symbol.Dropout(flat, p=0.0)
    # fc0 = mx.symbol.FullyConnected(data=flat, num_hidden=128, name='fc0')
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')

    # fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    result = mx.symbol.SoftmaxOutput(fc1, label, multi_output=True, normalization='batch', grad_scale=0.1)
    loss = multi_focalloss(result, label)
    # loss = mx.sym.softmax_cross_entropy(result, label)
    return mx.symbol.Group([mx.sym.BlockGrad(result), mx.sym.MakeLoss(loss)])

def resnet_3d_patch(units, num_stage, filter_list, num_class, bottle_neck=True,
           bn_mom=0.9, workspace=512, memonger=False):

    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    label = mx.sym.mean(label)
    # data = mx.sym.Cast(data=data, dtype=np.float16)

    data = mx.contrib.symbol.SyncBatchNorm(data=data, key='bn_data', fix_gamma=True, eps=2e-5,
                                           momentum=bn_mom, name='bn_data', use_global_stats=False)
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7,7,7), stride=(2,2,2), pad=(3,3,3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn0', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0', use_global_stats=False)
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3,3,3), stride=(2,2,2), pad=(1,1,1), pool_type='max')

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn1', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', use_global_stats=False)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7,7,7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    # flat = mx.symbol.Dropout(flat, p=0.5)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=10, name='fc1')
    fc1 = mx.symbol.Reshape(data=fc1, shape=(1, -1))
    fc1 = mx.symbol.Dropout(fc1, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=num_class, name='fc2')

    # fc2 = mx.sym.Cast(data=fc2, dtype=np.float32)
    result = mx.symbol.SoftmaxOutput(fc2, label, multi_output=True, normalization='batch', grad_scale=0.5)
    loss = mx.symbol.softmax_cross_entropy(fc1, label)
    # loss = multi_focalloss(result, label)
    return mx.symbol.Group([mx.sym.BlockGrad(result), mx.sym.BlockGrad(loss)])






























