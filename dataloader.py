#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:46:17 2020

@author: tangwen
"""

import mxnet as mx
import numpy as np
import cv2
import pydicom as dicom
import csv
import os
import random
import itertools
import scipy.ndimage
# from preprocessing import plot_3d
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

def plot_3d(array_3d, threshold=900):
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # p = array_3d.transpose(2,1,0)
    p = array_3d
    p = p[:,:,:]

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, p.shape[0])  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, p.shape[1])  # b = 10
    ax.set_zlim(0, p.shape[2])  # c = 16

    plt.show()

def read_csv(csvfile):
    lines = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            lines.append(row)
    return lines

def read_txt(txtfile):
    lines = []
    with open(txtfile, 'r') as file:
        lines = file.readlines()
    lines = [l.strip() for l in lines]
    return lines

def rotation(img, rotate_angle=90):
    angle_list = [i*rotate_angle for i in range(int(360/rotate_angle))]
    angle = angle_list[random.randint(0, int(360/rotate_angle)-1)]
    new_img = scipy.ndimage.rotate(img, angle, axes=(1, 2), reshape=False,
                         order=3, mode='constant', cval=0.0, prefilter=True)
    return new_img

def aug_swap(img):
    combine_list = [i for i in itertools.permutations([0, 1, 2], 3)]
    aug = combine_list[random.randint(0, len(combine_list)-1)]
    new_img = np.transpose(img, axes=aug)
    return new_img

def flip(img):
    flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
    new_img = np.ascontiguousarray(img[::flipid[0],::flipid[1],::flipid[2]])
    return new_img

def crop(img, crop_size=[0.6, 1.0]):
    length_ori = img.shape[0] / 2
    length_min = int(crop_size[0] * length_ori)
    length_max = int(crop_size[1] * length_ori)
    length_new = random.randint(length_min, length_max)

    center_min, center_max = length_new, img.shape[0] - length_new
    img_new = img[center_min-length_new:center_max+length_new, \
                  center_min-length_new:center_max+length_new, \
                  center_min-length_new:center_max+length_new]
    return img_new

def large_shift_crop(img, size=[0.5, 0.8]):
    length_ori = img.shape[0]
    center_ori = int(length_ori / 2)
    length_new_min = int(length_ori * size[0] / 2)
    length_new_max = int(length_ori * size[1] / 2)
    length_new = random.randint(length_new_min, length_new_max)

    center_min, center_max = length_new, img.shape[0] - length_new

    img_new = img[center_min-length_new:center_max+length_new, \
                  center_min-length_new:center_max+length_new, \
                  center_min-length_new:center_max+length_new]
    return img_new

def brightness(img, p=0.2):
    img = (img - (-1024)) / (2048 - (-1024)) * 255
    scale_1 = random.uniform(-p, p)
    img = img * (1 + scale_1)
    scale_2 = random.uniform(-p, p)
    img = img + 255 * scale_2
    img[img > 255] = 255
    img[img < 0] = 0
    return img
    
def resize(img, new_shape=[64,64,64]):
    ori_shape = img.shape
    resize_scale = np.array(new_shape) / ori_shape
    new_img = scipy.ndimage.zoom(img, resize_scale, mode='nearest', order=2)
    return new_img

def rotation_2(img1, img2, rotate_angle=90):
    angle_list = [i*rotate_angle for i in range(int(360/rotate_angle))]
    angle = angle_list[random.randint(0, int(360/rotate_angle)-1)]
    new_img1 = scipy.ndimage.rotate(img1, angle, axes=(1, 2), reshape=False,
                         order=3, mode='constant', cval=0.0, prefilter=True)
    new_img2 = scipy.ndimage.rotate(img2, angle, axes=(1, 2), reshape=False,
                         order=3, mode='constant', cval=0.0, prefilter=True)
    return new_img1, new_img2

def aug_swap_2(img1, img2):
    combine_list = [i for i in itertools.permutations([0, 1, 2], 3)]
    aug = combine_list[random.randint(0, len(combine_list)-1)]
    new_img1 = np.transpose(img1, axes=aug)
    new_img2 = np.transpose(img2, axes=aug)
    return new_img1, new_img2

def flip_2(img1, img2):
    flipid = np.array([1,np.random.randint(2),np.random.randint(2)])*2-1
    new_img1 = np.ascontiguousarray(img1[::flipid[0],::flipid[1],::flipid[2]])
    new_img2 = np.ascontiguousarray(img2[::flipid[0],::flipid[1],::flipid[2]])
    return new_img1, new_img2

def crop_2(img1, img2, crop_size=[0.6, 1.0]):
    length_ori = img1.shape[0] / 2
    length_min = int(crop_size[0] * length_ori)
    length_max = int(crop_size[1] * length_ori)
    length_new = random.randint(length_min, length_max)

    center_min, center_max = length_new, img1.shape[0] - length_new
    new_img1 = img1[center_min-length_new:center_max+length_new, \
                    center_min-length_new:center_max+length_new, \
                    center_min-length_new:center_max+length_new]
    new_img2 = img2[center_min-length_new:center_max+length_new, \
                    center_min-length_new:center_max+length_new, \
                    center_min-length_new:center_max+length_new]
    return new_img1, new_img2

def normalize_img(img1, img2):
    img1 = (img1 - (img1.min())) / (img1.max() - (img1.min())) * 255
    img2 = (img2 - (img2.min())) / (img2.max() - (img2.min())) * 255
    return img1, img2

def brightness_2(img1, img2, p=0.2):
    scale_1 = random.uniform(-p, p)
    img1 = img1 * (1 + scale_1)
    img2 = img2 * (1 + scale_1)
    scale_2 = random.uniform(-p, p)
    img1 = img1 + 255 * scale_2
    img2 = img2 + 255 * scale_2
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    img2[img2 > 255] = 255
    img2[img2 < 0] = 0
    return img1 / 255.0, img2 / 255.0

def brightness_3(img1, img2, p=0.2):
    scale_1 = random.uniform(-p, p)
    img1 = img1 * (1 + scale_1)
    img2 = img2 * (1 + scale_1)
    scale_2 = random.uniform(-p, p)
    img1 = img1 + (img1.max() - img1.min()) * scale_2
    img2 = img2 + (img2.max() - img2.min()) * scale_2

    return img1, img2

class CustomIter_1(mx.io.DataIter):
    def __init__(self, root, batch_size, pid_size, class_num, istrain=True, shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.pid_size = pid_size
        self.istrain = istrain
        self.shuffle = shuffle
        if istrain: self.txtfile = os.path.join(self.root, 'train.txt')
        else: self.txtfile = os.path.join(self.root, 'test.txt')
        self.pid_list_ori = read_txt(self.txtfile)
        self.class_num = class_num
        self.load_piddic()

        self.reset()

    def __iter__(self):
        return self

    def load_piddic(self):
        self.pid_dic = {}
        for pid in self.pid_list_ori:
            if pid not in self.pid_dic:
                self.pid_dic[pid] = []
            patch_list = os.listdir(os.path.join(self.root, 'patch_all', pid))
            zmax = max([int(p[:-4].split('_')[0]) for p in patch_list])
            ymax = max([int(p[:-4].split('_')[1]) for p in patch_list])
            xmax = max([int(p[:-4].split('_')[2]) for p in patch_list])
            for p in patch_list:
                if int(p[:-4].split('_')[0]) == 0 or int(p[:-4].split('_')[0]) == zmax or \
                   int(p[:-4].split('_')[1]) == 0 or int(p[:-4].split('_')[1]) == ymax or \
                   int(p[:-4].split('_')[2]) == 0 or int(p[:-4].split('_')[2]) == xmax:
                       continue
                self.pid_dic[pid].append(p)

    def balance_sample(self):
        count, label_list, self.pid_list = [0 for i in range(self.class_num)], [], []
        for pid in self.pid_list_ori:
            l = int(pid.split('_')[-1])
            label_list.append(l)
            count[l] += 1

        resample_time = [max(count[j]/count[i] for j in range(self.class_num)) for i in range(self.class_num)]
        resample_rate = [r - int(r) for r in resample_time]
        for i in range(len(self.pid_list_ori)):
            l = label_list[i]
            if random.random() <= resample_rate[l]:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]))
            else:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]) - 1)

            self.pid_list.append(self.pid_list_ori[i])

        return self.pid_list

    def reset(self):
        if self.istrain:
            if self.shuffle:
                random.shuffle(self.pid_list_ori)
                self.pid_list = self.balance_sample()
                random.shuffle(self.pid_list)
            else:
                self.pid_list = self.balance_sample()
        else:
            self.pid_list = (self.pid_list_ori).copy()

        self.cur_batch = 0
        self.batch_num = int(len(self.pid_list) / self.pid_size)

    def img_enhancement(self, img):
        probability = 0.7
        if self.istrain:
            if random.random() > probability:
                img = aug_swap(img)
            if random.random() > probability:
                img = flip(img)
            if random.random() > probability:
                img = rotation(img)
            # if random.random() > probability:
            #     img = crop(img)
            img = crop(img, crop_size=[0.75, 1.0])
            img = resize(img, new_shape=[64, 64, 64])
        else:
            img = crop(img, crop_size=[1.0, 1.0])
            img = resize(img, new_shape=[64, 64, 64])
        return img

    def get_patch(self, index):
        pid = self.pid_list[index]
        patch_list = self.pid_dic[pid]
        random.shuffle(patch_list)
        patches = []
        for i in range(self.batch_size):
            select_i = random.randint(0, len(patch_list)-1)
            path = os.path.join(self.root, 'patch_all', pid, patch_list[select_i])
            img = np.load(path)
            img = self.img_enhancement(img)
            patches.append(img[np.newaxis, :, :, :])
        return patches

    def get_batch(self):
        data, label = [], []
        for i in range(self.cur_batch * self.pid_size, (self.cur_batch+1) * self.pid_size):
            if not self.istrain:
                if i >= len(self.pid_list): i = i % len(self.pid_list)
            data += self.get_patch(i)
            label += [int(self.pid_list[i].split('_')[-1])] * self.batch_size

        data = mx.nd.array(np.array(data))
        label = mx.nd.array(np.array(label))

        return data, label

    def __next__(self):
        return self.next()
    @property
    def provide_data(self):
        return 'data'
    @property
    def provide_label(self):
        return 'label'

    def next(self):
        if self.cur_batch < self.batch_num:
            data, label = self.get_batch()
            if self.cur_batch % 2 == 1:
                print('processing on batch: %d'%self.cur_batch + '/%d'%self.batch_num)
            self.cur_batch += 1
            return mx.io.DataBatch([data], [label])
        else:
            raise StopIteration

class CustomIter_2(mx.io.DataIter):
    def __init__(self, root, batch_size, class_num, istrain=True, shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.istrain = istrain
        self.shuffle = shuffle
        if istrain: self.txtfile = os.path.join(self.root, 'train.txt')
        else: self.txtfile = os.path.join(self.root, 'test.txt')
        self.pid_list_ori = read_txt(self.txtfile)
        self.class_num = class_num

        self.reset()

    def __iter__(self):
        return self

    def balance_sample(self):
        count, label_list, self.pid_list = [0 for i in range(self.class_num)], [], []
        for pid in self.pid_list_ori:
            l = int(pid.split('_')[-1])
            label_list.append(l)
            count[l] += 1

        resample_time = [max(count[j]/count[i] for j in range(self.class_num)) for i in range(self.class_num)]
        resample_rate = [r - int(r) for r in resample_time]
        for i in range(len(self.pid_list_ori)):
            l = label_list[i]
            if random.random() <= resample_rate[l]:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]))
            else:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]) - 1)

            self.pid_list.append(self.pid_list_ori[i])

        return self.pid_list

    def reset(self):
        if self.istrain:
            if self.shuffle:
                random.shuffle(self.pid_list_ori)
                self.pid_list = self.balance_sample()
                random.shuffle(self.pid_list)
            else:
                self.pid_list = self.balance_sample()
        else:
            self.pid_list = (self.pid_list_ori).copy()

        self.cur_batch = 0
        self.batch_num = int(len(self.pid_list) / self.batch_size)

    def img_enhancement(self, img):
        probability = 0.7
        if self.istrain:
            if random.random() > probability:
                img = aug_swap(img)
            if random.random() > probability:
                img = flip(img)
            if random.random() > probability:
                img = rotation(img)
            # if random.random() > probability:
            #     img = crop(img)
            img = crop(img, crop_size=[1.0, 1.0])
            img = resize(img)
        else:
            img = crop(img, crop_size=[1.0, 1.0])
            img = resize(img)
        return img

    def helper(self, index, savepath):
        i = index % self.batch_size
        if not self.istrain:
            if index >= len(self.pid_list): index = index % len(self.pid_list)
        path = os.path.join(self.root, 'npy_all', self.pid_list[index] + '.npy')
        img = np.load(path)
        img = self.img_enhancement(img)
        np.save(os.path.join(savepath, str(i)+'.npy'), img[np.newaxis, :, :, :])
        int(self.pid_list[index].split('_')[-1])

    def get_batch(self):
        data, label = [], []
        savepath = '/media/tangwen/data/4D-classification/data/tmp'
        work_partial = partial(self.helper, savepath=savepath)
        pool = Pool(self.batch_size)
        pool.map(work_partial, range(self.cur_batch * self.batch_size, (self.cur_batch+1) * self.batch_size))
        pool.close()
        pool.join()

        # for i in range(self.cur_batch * self.batch_size, (self.cur_batch+1) * self.batch_size):
        #     if not self.istrain:
        #         if i >= len(self.pid_list): i = i % len(self.pid_list)
        #     path = os.path.join(self.root, 'npy', self.pid_list[i]+'.npy')
        #     img = np.load(path)
        #     img = self.img_enhancement(img)
        #     data.append(img[np.newaxis, :, :, :])
        #     label.append(int(self.pid_list[i].split('_')[-1]))

        for i in range(self.cur_batch * self.batch_size, (self.cur_batch+1) * self.batch_size):
            if not self.istrain:
                if i >= len(self.pid_list): i = i % len(self.pid_list)
            ii = i % self.batch_size
            label.append(int(self.pid_list[i].split('_')[-1]))
            # print(self.pid_list[i])
            img = np.load(os.path.join(savepath, str(ii)+'.npy'))
            data.append(img)
        data = mx.nd.array(np.array(data))
        label = mx.nd.array(np.array(label))
        return data, label

    def __next__(self):
        return self.next()
    @property
    def provide_data(self):
        return 'data'
    @property
    def provide_label(self):
        return 'label'

    def next(self):
        if self.cur_batch < self.batch_num:
            data, label = self.get_batch()
            if self.cur_batch % 2 == 1:
                print('processing on batch: %d'%self.cur_batch + '/%d'%self.batch_num)
            self.cur_batch += 1
            return mx.io.DataBatch([data], [label])
        else:
            raise StopIteration

class CustomIter_sufu(mx.io.DataIter):
    def __init__(self, root, batch_size, class_num, istrain=True, shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.istrain = istrain
        self.shuffle = shuffle
        if istrain: self.txtfile = os.path.join(self.root, 'train.txt')
        else: self.txtfile = os.path.join(self.root, 'test.txt')
        self.pid_list_ori = read_txt(self.txtfile)
        self.class_num = class_num

        self.reset()

    def __iter__(self):
        return self

    def balance_sample(self):
        count, label_list, self.pid_list = [0 for i in range(self.class_num)], [], []
        for pid in self.pid_list_ori:
            l = int(pid.split('_')[-1])
            label_list.append(l)
            count[l] += 1

        resample_time = [max(count[j]/count[i] for j in range(self.class_num)) for i in range(self.class_num)]
        # resample_time = [r*2 if r > 1 else r for r in resample_time]
        resample_rate = [r - int(r) for r in resample_time]
        for i in range(len(self.pid_list_ori)):
            l = label_list[i]
            if random.random() <= resample_rate[l]:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]))
            else:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]) - 1)

            self.pid_list.append(self.pid_list_ori[i])

        # random resample
        # tmp_pid_list = []
        # for pid in self.pid_list:
        #     if random.random() > 0.66:
        #         tmp_pid_list.append(pid)
        #         tmp_pid_list.append(pid)
        #     else:
        #         tmp_pid_list.append(pid)
        # self.pid_list = tmp_pid_list

        return self.pid_list

    def reset(self):
        if self.istrain:
            if self.shuffle:
                random.shuffle(self.pid_list_ori)
                self.pid_list = self.balance_sample()
                random.shuffle(self.pid_list)
            else:
                self.pid_list = self.balance_sample()
        else:
            self.pid_list = (self.pid_list_ori).copy()

        self.cur_batch = 0
        self.batch_num = int(len(self.pid_list) / self.batch_size)

    def img_enhancement(self, img_1st, img_3m):
        probability = 0.75
        if self.istrain:
            # img_1st, img_3m = normalize_img(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = aug_swap_2(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = flip_2(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = rotation_2(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = brightness_3(img_1st, img_3m, p=0.2)
            if random.random() > probability:
                img_1st, img_3m = crop_2(img_1st, img_3m, crop_size=[0.5, 1.0])
            img_1st = resize(img_1st)
            img_3m = resize(img_3m)
        else:
            # if random.random() > probability:
            #     img_1st, img_3m = aug_swap_2(img_1st, img_3m)
            # if random.random() > probability:
            #     img_1st, img_3m = flip_2(img_1st, img_3m)
            # if random.random() > probability:
            #     img_1st, img_3m = rotation_2(img_1st, img_3m)
            # if random.random() > probability:
            #     img_1st, img_3m = brightness_3(img_1st, img_3m, p=0.2)
            # if random.random() > probability:
            #     img_1st, img_3m = crop_2(img_1st, img_3m, crop_size=[0.5, 1.0])
            img_1st, img_3m = crop_2(img_1st, img_3m, crop_size=[1.0, 1.0])
            img_1st = resize(img_1st)
            img_3m = resize(img_3m)
        return img_1st, img_3m

    def get_batch(self):
        data, label = [], []
        for i in range(self.cur_batch * self.batch_size, (self.cur_batch+1) * self.batch_size):
            if True:
                if i >= len(self.pid_list): i = i % len(self.pid_list)

            path_1st = os.path.join(self.root, 'npy_detection', self.pid_list[i]+'_1st.npy')
            path_3m = os.path.join(self.root, 'npy_detection', self.pid_list[i]+'_3m.npy')
            img_1st = np.load(path_1st).astype(float)
            img_3m = np.load(path_3m).astype(float)
            img_1st, img_3m = self.img_enhancement(img_1st, img_3m)
            img = np.concatenate((img_1st[np.newaxis,:,:,:], img_3m[np.newaxis,:,:,:]), axis=0)
            data.append(img)
            label.append(int(self.pid_list[i].split('_')[-1]))

        data = mx.nd.array(np.array(data))
        label = mx.nd.array(np.array(label)[:, np.newaxis])
        return data, label

    def __next__(self):
        return self.next()
    @property
    def provide_data(self):
        return 'data'
    @property
    def provide_label(self):
        return 'label'

    def next(self):
        if self.cur_batch <= self.batch_num:
            data, label = self.get_batch()
            if self.cur_batch % 2 == 1:
                print('processing on batch: %d'%self.cur_batch + '/%d'%self.batch_num)
            self.cur_batch += 1
            return mx.io.DataBatch([data], [label])
        else:
            raise StopIteration

class CustomIter_sufu2(mx.io.DataIter):
    def __init__(self, root, batch_size, class_num, istrain=True, shuffle=True):
        self.root = root
        self.batch_size = batch_size
        self.istrain = istrain
        self.shuffle = shuffle
        if istrain: self.txtfile = os.path.join(self.root, 'train.txt')
        else: self.txtfile = os.path.join(self.root, 'test.txt')
        self.pid_list_ori = read_txt(self.txtfile)
        self.class_num = class_num

        self.reset()

    def __iter__(self):
        return self

    def balance_sample(self):
        count, label_list, self.pid_list = [0 for i in range(self.class_num)], [], []
        for pid in self.pid_list_ori:
            l = int(pid.split('_')[-1])
            label_list.append(l)
            count[l] += 1

        resample_time = [max(count[j]/count[i] for j in range(self.class_num)) for i in range(self.class_num)]
        # resample_time = [r*2 if r > 1 else r for r in resample_time]
        resample_rate = [r - int(r) for r in resample_time]
        for i in range(len(self.pid_list_ori)):
            l = label_list[i]
            if random.random() <= resample_rate[l]:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]))
            else:
                self.pid_list += [self.pid_list_ori[i]] * (int(resample_time[l]) - 1)

            self.pid_list.append(self.pid_list_ori[i])

        return self.pid_list

    def reset(self):
        if self.istrain:
            if self.shuffle:
                random.shuffle(self.pid_list_ori)
                self.pid_list = self.balance_sample()
                random.shuffle(self.pid_list)
            else:
                self.pid_list = self.balance_sample()
        else:
            self.pid_list = (self.pid_list_ori).copy()

        self.cur_batch = 0
        self.batch_num = int(len(self.pid_list) / self.batch_size)

    def img_enhancement(self, img_1st, img_3m):
        probability = 0.75
        if self.istrain:
            # img_1st, img_3m = normalize_img(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = aug_swap_2(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = flip_2(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = rotation_2(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = brightness_3(img_1st, img_3m)
            if random.random() > probability:
                img_1st, img_3m = crop_2(img_1st, img_3m, crop_size=[0.5, 1.0])
            img_1st = resize(img_1st)
            img_3m = resize(img_3m)
        else:
            # img_1st, img_3m = normalize_img(img_1st, img_3m)
            img_1st, img_3m = crop_2(img_1st, img_3m, crop_size=[1.0, 1.0])
            img_1st = resize(img_1st)
            img_3m = resize(img_3m)
        return img_1st, img_3m

    def get_batch(self):
        data, label = [], []
        for i in range(self.cur_batch * self.batch_size, (self.cur_batch+1) * self.batch_size):
            if True:
                if i >= len(self.pid_list): i = i % len(self.pid_list)

            path_1st = os.path.join(self.root, 'npy_segmentation', self.pid_list[i]+'_1st.npy')
            path_3m = os.path.join(self.root, 'npy_segmentation', self.pid_list[i]+'_3m.npy')
            img_1st = np.load(path_1st).astype(float)
            img_3m = np.load(path_3m).astype(float)
            img_1st, img_3m = self.img_enhancement(img_1st, img_3m)
            # img = img_3m[np.newaxis,:,:,:]
            img = np.concatenate((img_1st[np.newaxis, :, :, :], img_3m[np.newaxis, :, :, :]), axis=0)
            data.append(img)
            label.append(int(self.pid_list[i].split('_')[-1]))

        data = mx.nd.array(np.array(data))
        label = mx.nd.array(np.array(label)[:, np.newaxis])
        return data, label

    def __next__(self):
        return self.next()
    @property
    def provide_data(self):
        return 'data'
    @property
    def provide_label(self):
        return 'label'

    def next(self):
        if self.cur_batch <= self.batch_num:
            data, label = self.get_batch()
            if self.cur_batch % 2 == 1:
                print('processing on batch: %d'%self.cur_batch + '/%d'%self.batch_num)
            self.cur_batch += 1
            return mx.io.DataBatch([data], [label])
        else:
            raise StopIteration

if __name__ == '__main__':
    root = '/media/tangwen/data/4D-classification/data/'
    train_iter = CustomIter_sufu(root=root, batch_size=12, class_num=2, istrain=True, shuffle=True)
    b = train_iter.get_batch()
    num = 1
    data, label = b[0][num, :, :, :, :], b[1][num]
    data = data.asnumpy()
    label = label.asnumpy()

    # root = '/media/tangwen/data/4D-classification/data/'
    # train_iter = CustomIter_1(root=root, batch_size=64, pid_size=4, class_num=3, istrain=True, shuffle=True)
    # b = train_iter.get_batch()
    # num = 1
    # data, label = b[0][num, :, :, :, :], b[1][num]
    # data = data.asnumpy()
    # label = label.asnumpy()















