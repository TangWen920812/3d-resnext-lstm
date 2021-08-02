#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:30:30 2020

@author: tangwen
"""


import mxnet as mx
from resnet import resnet_3d
from resnext import resnext_3d_rnn
import numpy as np
from dataloader import CustomIter_sufu
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

net_depth = 34
n_classes = 2
ctx = [mx.gpu(i) for i in [1]]
pretrain = False
resume = True

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
symbol = resnext_3d_rnn(units=units, num_stage=4, filter_list=filter_list, num_class=n_classes, bottle_neck=bottle_neck,
           bn_mom=0.9, workspace=512, memonger=False, ratio_list=ratio_list)
name_sym = symbol.get_internals()
classifier = mx.mod.Module(symbol, context=ctx, data_names=['data'], label_names=['label'])
classifier.bind(data_shapes=[['data', (batch_size, 2, img_size, img_size, img_size)]],
                              label_shapes=[['label', (batch_size, 1)],], for_training=False)

##################################################################################
root = '/media/tangwen/data/4D-classification/data'
# train_iter = CustomIter_sufu(root=root, batch_size=batch_size, class_num=n_classes, istrain=True, shuffle=True)
test_iter = CustomIter_sufu(root=root, batch_size=batch_size, class_num=n_classes, istrain=False, shuffle=False)

if resume:
    start_e = 90
    symbol, arg_params, aux_params = mx.model.load_checkpoint('/media/tangwen/data/4D-classification/model/resnext34_rnn', start_e)
    classifier.set_params(arg_params, aux_params)
    # classifier.init_params(mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=3))
else:
    start_e = 0
    classifier.init_params(mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=3))

# mx.viz.plot_network(symbol,title='resnet',save_format='pdf',hide_weights=True,
#                     shape={'data':(1,1,64,64,64), 'txt_data':(1,19), 'label':(1,1)}).view()
classifier.init_optimizer(optimizer = 'adam', optimizer_params=(
                                               ('learning_rate', 1E-4),
                                               ('beta1', 0.9),
                                               ('beta2', 0.999)
                                               ))


def plot_auc(fpr, tpr, roc_auc, i):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    # plt.savefig('./result/' + 'result_rnn_seg.png')

def metric_acc(predict, label):
    try:
        predict = predict.asnumpy()
        label = label.asnumpy()
    except:
        predict = np.array(predict)
        label = np.array(label)
    # predict_index = np.argmax(predict, axis=1)
    predict[np.where(predict < 0.5)] = 0
    acc = np.sum(predict == label) / label.shape[0]
    return acc

def metric_auc(predict, label):
    predict = np.array(predict)
    label = label
    auc_result = []
    for i in range(n_classes):
        prob = predict[:, i]
        auc_v = roc_auc_score([1 if l==i else 0 for l in label], prob)
        l = [1 if l==i else 0 for l in label]
        fpr, tpr, thresholds = roc_curve(l, prob)
        roc_auc = auc(fpr, tpr)
        plot_auc(fpr, tpr, roc_auc, i)
        auc_result.append(auc_v)
    return auc_result

count = 2
mean_loss, mean_acc, batch_mean = [], [], [0, 0]
predict_all_acc, predict_all_auc, label_all = [], [], []
batch_num = -1
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
    predict_all += list(predict.asnumpy()[:, 0])
    label_all += list(label.asnumpy()[:, 0])

    mean_acc.append(acc)
    mean_loss.append(np.mean(loss.asnumpy()))

auc_score = roc_auc_score(label_all[0:123], predict_all[0:123])
fpr, tpr, thresholds = roc_curve(label_all[0:123], predict_all[0:123])
roc_auc = auc(fpr, tpr)
plot_auc(fpr, tpr, roc_auc, 5)
# acc = accuracy_score(label_all[0:45], predict_all[0:45])
print('auc: ', auc_score)

import csv
with open('./result/train.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['pred', 'label'])
    for i in range(len(label_all[0:45])):
        writer.writerow([predict_all[i], label_all[i]])

















