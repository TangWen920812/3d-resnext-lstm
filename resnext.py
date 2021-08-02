#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:09:38 2020

@author: tangwen
"""
import mxnet as mx
import numpy as np

sbn = False

def multi_focalloss(predict, label, gamma=2):
    label_onehot = mx.symbol.one_hot(label, depth=4)
    focal_loss = -(1 - predict * label_onehot)**gamma * mx.symbol.log10(predict + 1e-7)
    return mx.symbol.mean(focal_loss)

def focalloss(predict, label, gamma=2, alpha=0.5):
    ce_loss = - alpha * (1-predict)**gamma * label * mx.sym.log10(predict + 1e-7) \
                - (1-alpha) * predict**gamma * (1 - label) * mx.sym.log10(1 - predict + 1e-7)
    weight_ce_loss = mx.sym.mean(ce_loss)
    return weight_ce_loss

def residual_unit(data, num_filter, ratio, stride, dim_match, name, num_group,
                  bottle_neck=True,  bn_mom=0.9, workspace=2048, memonger=False):
    """Return ResNext Unit symbol for building ResNext
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    bottle_neck : Boolen
        Whether or not to adopt bottle_neck trick as did in ResNet
    num_group : int
        Number of convolution groupes
    bn_mom : float
        Momentum of batch normalization
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.5), kernel=(1,1,1), stride=(1,1,1), pad=(0,0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        if sbn:
            bn1 = mx.contrib.symbol.SyncBatchNorm(data=conv1, key=name + '_bn1', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')

        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3,3), stride=stride, pad=(1,1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if sbn:
            bn2 = mx.contrib.symbol.SyncBatchNorm(data=conv2, key=name + '_bn2', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1,1), stride=(1,1,1), pad=(0,0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if sbn:
            bn3 = mx.contrib.symbol.SyncBatchNorm(data=conv3, key=name + '_bn3', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        else:
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        squeeze = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7, 7), pool_type='avg', name=name + '_squeeze')
        squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        bn3 = mx.symbol.broadcast_mul(bn3, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1, 1)))

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            shortcut = mx.contrib.symbol.SyncBatchNorm(data=shortcut_conv, key=name + '_sc_bn', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise =  bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3,3), stride=stride, pad=(1,1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        if sbn:
            bn1 = mx.contrib.symbol.SyncBatchNorm(data=conv1, key=name + '_bn1', fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        else:
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3,3), stride=(1,1,1), pad=(1,1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if sbn:
            bn2 = mx.contrib.symbol.SyncBatchNorm(data=conv2, key=name + '_bn2', fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        else:
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        squeeze = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7, 7), pool_type='avg', name=name + '_squeeze')
        squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
        excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
        excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
        excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
        excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
        bn2 = mx.symbol.broadcast_mul(bn2, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1, 1)))

        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            if sbn:
                shortcut = mx.contrib.symbol.SyncBatchNorm(data=shortcut_conv, key=name + '_sc_bn', fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
            else:
                shortcut = mx.symbol.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')


def resnext_3d(units, num_stage, filter_list, num_class, ratio_list, num_group=8, drop_out=0.0,
            bottle_neck=True, bn_mom=0.9, workspace=2048, memonger=False):
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Output size of symbol
    num_groupes: int
		Number of convolution groups
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    data_type : str
        Dataset type, only cifar10, imagenet and vggface supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    # data = mx.sym.Cast(data=data, dtype=np.float16)
    if sbn:
        data = mx.contrib.symbol.SyncBatchNorm(data=data, key='bn_data', fix_gamma=True, eps=2e-5, momentum=bn_mom,
                                           name='bn_data', use_global_stats=False)
    else:
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', use_global_stats=False)
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7, 7), stride=(2,2,2), pad=(3, 3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    if sbn:
        body = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn0', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    else:
        body = mx.symbol.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3, 3), stride=(2,2,2), pad=(1,1,1), pool_type='max')

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], ratio_list[2], (1 if i==0 else 2, 1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bottle_neck=bottle_neck,  
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)

        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], ratio_list[2], (1,1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 num_group=num_group, bottle_neck=bottle_neck, bn_mom=bn_mom, workspace=workspace, memonger=memonger)

    if sbn:
        bn1 = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn1', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', use_global_stats=False)
    else:
        bn1 = mx.symbol.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', use_global_stats=False)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7,7,7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    flat = mx.symbol.Dropout(flat, p=0.5, mode='training')
    flat = mx.symbol.FullyConnected(data=flat, num_hidden=256, name='fc01')
    flat = mx.symbol.Dropout(flat, p=0.5, mode='training')
    flat = mx.symbol.FullyConnected(data=flat, num_hidden=128, name='fc02')
    flat = mx.symbol.Dropout(flat, p=0.5, mode='training')
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class-1, name='fc1')
    # fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)

    # result = mx.symbol.SoftmaxOutput(fc1, label, multi_output=True, normalization='batch', grad_scale=0.0)
    # loss = mx.symbol.softmax_cross_entropy(fc1, label)
    result = mx.sym.sigmoid(fc1)
    loss = focalloss(result, label)
    return mx.symbol.Group([mx.sym.BlockGrad(result), mx.sym.MakeLoss(loss)])

def resnext_3d_patch(units, num_stage, filter_list, num_class, ratio_list, num_group=8, drop_out=0.0,
            bottle_neck=True, bn_mom=0.9, workspace=2048, memonger=False):
    """Return ResNeXt symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Output size of symbol
    num_groupes: int
		Number of convolution groups
    drop_out : float
        Probability of an element to be zeroed. Default = 0.0
    data_type : str
        Dataset type, only cifar10, imagenet and vggface supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    # data = mx.sym.Cast(data=data, dtype=np.float16)
    label = mx.sym.mean(label)

    data = mx.contrib.symbol.SyncBatchNorm(data=data, key='bn_data', fix_gamma=True, eps=2e-5, momentum=bn_mom,
                                           name='bn_data', use_global_stats=False)
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7, 7), stride=(2,2,2), pad=(3, 3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn0', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3, 3), stride=(2,2,2), pad=(1,1,1), pool_type='max')

    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], ratio_list[2], (1 if i==0 else 2, 1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bottle_neck=bottle_neck,
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)

        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], ratio_list[2], (1,1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 num_group=num_group, bottle_neck=bottle_neck, bn_mom=bn_mom, workspace=workspace, memonger=memonger)

    bn1 = mx.contrib.symbol.SyncBatchNorm(data=body, key='bn1', fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', use_global_stats=False)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7,7,7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    # flat = mx.symbol.Dropout(flat, p=0.5)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=10, name='fc1')
    fc1 = mx.symbol.Reshape(data=fc1, shape=(1, -1))
    fc1 = mx.symbol.Dropout(fc1, p=0.0)
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=num_class, name='fc2')

    # fc2 = mx.sym.Cast(data=fc2, dtype=np.float32)
    result = mx.symbol.SoftmaxOutput(fc2, label, multi_output=True, normalization='batch', grad_scale=0.5)
    loss = mx.symbol.softmax_cross_entropy(fc1, label)
    # loss = multi_focalloss(result, label)
    return mx.symbol.Group([mx.sym.BlockGrad(result), mx.sym.BlockGrad(loss*0.5)])

def resnext_3d_rnn(units, num_stage, filter_list, num_class, ratio_list, num_group=8, drop_out=0.0,
            bottle_neck=True, bn_mom=0.9, workspace=2048, memonger=False):
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    data1 = mx.sym.slice_axis(data, axis=1, begin=0, end=1)
    data2 = mx.sym.slice_axis(data, axis=1, begin=1, end=2)
    data_new = mx.sym.Concat(data1, data2, dim=0)

    stack = mx.rnn.SequentialRNNCell()
    for i in range(3):
        stack.add(mx.rnn.LSTMCell(num_hidden=512, prefix='lstm_l%d_' % i))
        stack.add(mx.rnn.DropoutCell(dropout=0.8, prefix='dropout_l%d_' % i))

    # data1
    data_new = mx.sym.BatchNorm(data=data_new, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data', use_global_stats=False)
    body = mx.sym.Convolution(data=data_new, num_filter=filter_list[0], kernel=(7, 7, 7), stride=(2,2,2), pad=(3, 3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.symbol.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3, 3), stride=(2,2,2), pad=(1,1,1), pool_type='max')
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], ratio_list[2], (1 if i==0 else 2, 1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), num_group=num_group, bottle_neck=bottle_neck,
                             bn_mom=bn_mom, workspace=workspace, memonger=memonger)

        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], ratio_list[2], (1,1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 num_group=num_group, bottle_neck=bottle_neck, bn_mom=bn_mom, workspace=workspace, memonger=memonger)
    bn1 = mx.symbol.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1', use_global_stats=False)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7,7,7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)

    flat1 = mx.sym.slice_axis(flat, axis=0, begin=0, end=24)
    flat2 = mx.sym.slice_axis(flat, axis=0, begin=24, end=48)
    flat1 = mx.sym.reshape(flat1, (0, 1, -1))
    flat2 = mx.sym.reshape(flat2, (0, 1, -1))
    flat = mx.sym.Concat(flat1, flat2, dim=1)
    stack.reset()
    flat, states = stack.unroll(2, inputs=flat, merge_outputs=True)

    flat = mx.symbol.Dropout(flat, p=0.5, mode='training')
    flat = mx.symbol.FullyConnected(data=flat, num_hidden=256, name='fc01')
    flat = mx.symbol.Dropout(flat, p=0.5, mode='training')
    flat = mx.symbol.FullyConnected(data=flat, num_hidden=128, name='fc02')
    flat = mx.symbol.Dropout(flat, p=0.5, mode='training')
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class-1, name='fc1')
    # fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)

    result = mx.sym.sigmoid(fc1)
    loss = focalloss(result, label)
    return mx.symbol.Group([mx.sym.BlockGrad(result), mx.sym.MakeLoss(loss)])
    # return mx.sym.MakeLoss(loss)























