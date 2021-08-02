#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:59:37 2020

@author: tangwen
"""

import mxnet as mx
from resnet import resnet_3d
from resnext import resnext_3d
import numpy as np
from dataloader import CustomIter_sufu
from sklearn.metrics import roc_auc_score, accuracy_score
import os

net_depth = 34
n_classes = 2
ctx = [mx.gpu(i) for i in [1]]
pretrain = False
resume = False

batch_size = 24
img_size = 64

if net_depth == 18:
    units = [2, 2, 2, 2]
elif net_depth == 34:
    units = [3, 4, 6, 3]
elif net_depth == 50:
    units = [3, 4, 6, 3]
elif net_depth == 101:
    units = [3, 4, 23, 3]
elif net_depth == 152:
    units = [3, 8, 36, 3]
elif net_depth == 200:
    units = [3, 24, 36, 3]
elif net_depth == 269:
    units = [3, 30, 48, 8]
else:
    raise ValueError("no experiments done on detph {}, you can do it youself".format(net_depth))

if net_depth >= 50:
    filter_list = [64, 256, 512, 1024, 2048]
    bottle_neck = True
else:
    filter_list = [64, 64, 128, 256, 512]
    bottle_neck = False
ratio_list = [0.25, 0.125, 0.0625, 0.03125]

##################################################################################
# symbol = resnet_3d(units=units, num_stage=4, filter_list=filter_list, num_class=n_classes, bottle_neck=bottle_neck,
#            bn_mom=0.9, workspace=512, memonger=False)
symbol = resnext_3d(units=units, num_stage=4, filter_list=filter_list, num_class=n_classes, bottle_neck=bottle_neck,
           bn_mom=0.9, workspace=512, memonger=False, ratio_list=ratio_list)
name_sym = symbol.get_internals()
classifier = mx.mod.Module(symbol, context=ctx, data_names=['data'], label_names=['label'])
classifier.bind(data_shapes=[['data', (batch_size, 2, img_size, img_size, img_size)]],
                              label_shapes=[['label', (batch_size, 1)],])

##################################################################################
root = '/media/tangwen/data/4D-classification/data'
train_iter = CustomIter_sufu(root=root, batch_size=batch_size, class_num=n_classes, istrain=True, shuffle=True)
test_iter = CustomIter_sufu(root=root, batch_size=batch_size, class_num=n_classes, istrain=False, shuffle=False)

if resume:
    start_e = 0
    symbol, arg_params, aux_params = mx.model.load_checkpoint('/media/tangwen/data/4D-classification/model/resnext34_m-pretrain', start_e)
    classifier.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    classifier.init_params(mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=3))
else:
    start_e = 0
    classifier.init_params(mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=3))

# mx.viz.plot_network(symbol,title='resnet',save_format='pdf',hide_weights=True,
#                     shape={'data':(1,1,64,64,64), 'txt_data':(1,19), 'label':(1,1)}).view()
classifier.init_optimizer(optimizer = 'adam', optimizer_params=(
                                               ('learning_rate', 1E-4),
                                               ('beta1', 0.9),
                                               ('beta2', 0.99),
                                               ('multi_precision', False)
                                               ))
def metric_acc(predict, label):
    predict = predict.asnumpy()
    label = label.asnumpy()
    # predict_index = np.argmax(predict, axis=1)
    predict[np.where(predict < 0.5)] = 0
    predict[np.where(predict >= 0.5)] = 1
    acc = np.sum(predict == label) / label.shape[0]
    return acc

def test_epoch(classifier, test_iter):
    mean_loss, mean_acc = [], []
    predict_all, label_all = [], []
    batch_num = -1
    while True:
        batch_num += 1
        try:
            batch = next(test_iter)
        except StopIteration:
            test_iter.reset()
            break

        classifier.forward(batch, is_train=False)
        batch_out = classifier.get_outputs()

        predict = batch_out[0]
        loss = batch_out[1]
        label = batch.label[0]
        acc = metric_acc(predict, label)
        predict_all += list(predict.asnumpy()[:,0])
        label_all += list(label.asnumpy()[:,0])
        
        mean_acc.append(acc)
        mean_loss.append(np.mean(loss.asnumpy()))
    print('test on epoch %d: '%e, '\nloss: ', sum(mean_loss)/len(mean_loss), 'acc: ',
      sum(mean_acc)/len(mean_acc))
    
    auc = roc_auc_score(label_all[0:45], predict_all[0:45])
    print('auc: ', auc)
    return auc
    
epochs = 200
count = 2
mean_loss, mean_acc, batch_mean = [], [], [0, 0]
predict_all, label_all = [], []
auc_best = 0
for e in range(start_e + 1, epochs):
    batch_num = -1
    while True:
        batch_num += 1
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter.reset()
            break

        classifier.forward_backward(batch)
        batch_out = classifier.get_outputs()
        classifier.update()

        predict = batch_out[0]
        loss = batch_out[1]
        label = batch.label[0]
        acc = metric_acc(predict, label)

        mean_acc.append(acc)
        mean_loss.append(np.mean(loss.asnumpy()))
        batch_mean[0] += np.mean(loss.asnumpy())
        batch_mean[1] += acc

        predict_all += list(predict.asnumpy()[:, 0])
        label_all += list(label.asnumpy()[:, 0])

        if batch_num % count == count - 1:
            print('loss: ', batch_mean[0]/count, 'acc: ', batch_mean[1]/count)
            batch_mean = [0, 0]

    train_auc = roc_auc_score(label_all, predict_all)
    print('epoch %d: '%e, '\n loss: ', sum(mean_loss)/len(mean_loss), 'acc: ',
          sum(mean_acc)/len(mean_acc), 'auc: ', train_auc)
    mean_loss, mean_acc = [], []
    predict_all, label_all = [], []
    batch_mean = [0, 0]

    classifier.save_checkpoint('/media/tangwen/data/4D-classification/model/resnext34_m', e)
    _, arg_params, aux_params = mx.model.load_checkpoint('/media/tangwen/data/4D-classification/model/resnext34_m', e)
    classifier.set_params(arg_params, aux_params)
    auc = test_epoch(classifier, test_iter)
    if auc > auc_best:
        auc_best = auc
        print('best epoch on: ', e)
    else:
        os.remove('/media/tangwen/data/4D-classification/model/resnext34_m' + '-%04d.params'%e)
    
print(auc_best)































