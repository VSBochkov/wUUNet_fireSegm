import os
import shutil
from distutils.dir_util import copy_tree
import math
from os.path import join

import cv2
import numpy as np
import torch
from albumentations import (
    Blur, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
    RandomBrightnessContrast, IAASharpen, IAAEmboss, OneOf, Compose
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as Pytorch_Dataset

from utils import (
    get_multiclass_label_map, get_binary_label_map
)

from zipfile import ZipFile

N_CLASS = 3  # 1
IMG_CHANNELS_NUM = 3


def _augm(p=.5):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.33),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.33),
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.33),
    ], p=p)


augm_train = _augm(1)
augm_val = _augm(1)


class MulticlassFiresegmDataset(Pytorch_Dataset):
    def __init__(self, base='train', transforms=None, is_for_training=True, win_divisor=1,
                 is_var_step=False, input_size=(224, 224), full_size=True):
        super(MulticlassFiresegmDataset, self).__init__()
        self.input_size = input_size
        if full_size:
            dataset_variant = 'fs_' + str(input_size[0])
        else:
            dataset_variant = 'ow_' + str(input_size[0])
        self.data_anim_path = join('dataset', join(dataset_variant, base))
        self.images_path = self.data_anim_path + '/images/'
        self.labels_path = self.data_anim_path + '/labels/'
        self.transforms = transforms
        self.win_divisor = win_divisor
        self.training = is_for_training
        self.var_step = is_var_step
        self.data_list = self.init_data_list()
        self.data_map, self.label_map = self.get_data_label_map()

    def init_data_list(self):
        data_list = []
        for video_num in range(0, len(os.listdir(self.images_path))):
            video_path = join(self.images_path, str(video_num))
            labels_video_path = join(self.labels_path, str(video_num))
            for sample_num in range(0, len(os.listdir(video_path))):
                label = np.load(join(labels_video_path, '{}.npy'.format(sample_num)))
                if label.shape[0] > self.input_size[0]:
                    if self.var_step:
                        bboxes = self.get_test_bboxes_var_step(label)
                    elif self.training:
                        bboxes = self.get_grid_bboxes(label)
                    else:
                        bboxes = self.get_test_bboxes_win_step(label)
                elif label.shape[1] > self.input_size[1]:
                    if self.var_step:
                        bboxes = self.get_test_row_bboxes_var_step(label)
                    elif self.training:
                        bboxes = self.get_row_bboxes(label)
                    else:
                        bboxes = self.get_test_row_bboxes(label)
                else:
                    bboxes = [(0, 0, self.input_size[0], self.input_size[1])]
                data_list.extend([
                    (video_num, sample_num, (i, j, h, w))
                    for i, j, h, w in bboxes])
        return data_list

    def get_data_label_map(self):
        data_map = {}
        labels_map = {}
        for video_num in range(0, len(os.listdir(self.images_path))):
            images_path = join(self.images_path, str(video_num))
            labels_path = join(self.labels_path, str(video_num))
            for sample_num in range(0, len(os.listdir(images_path))):
                img_path = join(images_path, '{}.jpg'.format(sample_num))
                lbl_path = join(labels_path, '{}.npy'.format(sample_num))
                label = np.load(lbl_path)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                data_map[(video_num, sample_num)] = img.copy()
                labels_map[(video_num, sample_num)] = label.copy()
        return data_map, labels_map

    def get_grid_bboxes(self, matrix):
        bboxes = []
        i, j = 0, 0
        imh, imw = matrix.shape[0], matrix.shape[1]
        winh, winw = self.input_size

        while i + winh <= imh:
            j = 0
            while j + winw <= imw:
                bboxes.append((i, j, winh, winw))
                j += int((imw - winw) / 4)
            i += int((imh - winh) / math.ceil(float(imh) / winh))

        return bboxes

    def get_test_bboxes_win_step(self, matrix):
        bboxes = []
        i, j = 0, 0
        imh, imw = matrix.shape[0], matrix.shape[1]
        winh, winw = self.input_size

        step = winw / self.win_divisor if self.win_divisor > 1 else 0
        while i + step < imh:
            j = 0
            while j + step < imw:
                bboxes.append((i, j, winh, winw))
                j += int(winw / self.win_divisor)
            i += int(winh / self.win_divisor)
        return bboxes

    def get_test_bboxes_var_step(self, matrix):
        bboxes = []
        i, j = 0, 0
        imh, imw = matrix.shape[0], matrix.shape[1]
        winh, winw = self.input_size

        dy = math.ceil((imh - winh) / math.floor(float(imh) / winh))
        dx = math.ceil((imw - winw) / math.floor(float(imw) / winw))
        while i + dy <= imh:
            j = 0
            while j + dx <= imw:
                bboxes.append((i, j, winh, winw))
                j += dx
            i += dy
        return bboxes

    def get_row_bboxes(self, matrix):
        bboxes = []
        i, j = 0, 0
        imh, imw = matrix.shape[0], matrix.shape[1]
        winh, winw = self.input_size

        while j + winw <= imw:
            bboxes.append((i, j, winh, winw))
            j += int((imw - winw) / 2)

        return bboxes

    def get_test_row_bboxes(self, matrix):
        bboxes = []
        i, j = 0, 0
        imh, imw = matrix.shape[0], matrix.shape[1]
        winh, winw = self.input_size

        while j < imw:
            bboxes.append((i, j, winh, winw))
            j += int(winw / self.win_divisor)

        return bboxes

    def get_test_row_bboxes_var_step(self, matrix):
        bboxes = []
        i, j = 0, 0
        imh, imw = matrix.shape[0], matrix.shape[1]
        winh, winw = self.input_size
        dx = math.ceil((imw - winw) / self.win_divisor)

        while j + winw <= imw:
            bboxes.append((i, j, winh, winw))
            j += dx

        return bboxes

    def __getitem__(self, idx):
        video_num, sample_num, bbox = self.data_list[idx]
        i, j, winh, winw = bbox
        imh, imw, _ = self.data_map[(video_num, sample_num)].shape
        image = np.zeros((winh, winw, IMG_CHANNELS_NUM), dtype=np.uint8)
        label = np.zeros((winh, winw), dtype=np.uint8)
        if imh < i + winh and imw < j + winw:
            image[: imh - i, : imw - j] = self.data_map[(video_num, sample_num)][i: imh, j: imw].copy()
            label[: imh - i, : imw - j] = self.label_map[(video_num, sample_num)][i: imh, j: imw].copy()
        elif imh < i + winh:
            image[: imh - i, :] = self.data_map[(video_num, sample_num)][i: imh, j: j + winw].copy()
            label[: imh - i, :] = self.label_map[(video_num, sample_num)][i: imh, j: j + winw].copy()
        elif imw < j + winw:
            image[:, : imw - j] = self.data_map[(video_num, sample_num)][i: i + winh, j: imw].copy()
            label[:, : imw - j] = self.label_map[(video_num, sample_num)][i: i + winh, j: imw].copy()
        else:
            image = self.data_map[(video_num, sample_num)][i: i + winh, j: j + winw].copy()
            label = self.label_map[(video_num, sample_num)][i: i + winh, j: j + winw].copy()
        if N_CLASS > 1:
            label = get_multiclass_label_map(label, N_CLASS)
        else:
            label = get_binary_label_map(label)
        image = np.asarray(image, np.float32)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        return {
            'img': np.transpose(image.copy(), (2, 0, 1)) / 255.0,
            'lbl': label,
            'sample_num_path': (video_num, sample_num),
            'bbox': bbox,
            'frame_resolution': (imh, imw)
        }

    def __len__(self):
        return len(self.data_list)

    def get_dataloader(self, batch_size=1, shuffle=False, num_workers=1):
        data_loader = DataLoader(self, batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 shuffle=shuffle,
                                 num_workers=num_workers)
        return data_loader


def collate_fn(batch):
    imgs = []
    mult_lbls = []
    sample_num_paths = []
    bboxes = []
    frame_resolutions = []

    for x in batch:
        imgs.append(x['img'])
        mult_lbls.append(x['lbl'])
        sample_num_paths.append(x['sample_num_path'])
        bboxes.append(x['bbox'])
        frame_resolutions.append(x['frame_resolution'])

    imgs = np.array(imgs)
    mult_lbls = np.asarray(mult_lbls)
    sample_num_paths = np.asarray(sample_num_paths)
    bboxes = np.asarray(bboxes)
    frame_resolutions = np.asarray(frame_resolutions)

    return {
        'imgs': torch.from_numpy(imgs),
        'lbls': torch.from_numpy(mult_lbls),
        'sample_num_paths': torch.from_numpy(sample_num_paths),
        'bboxes': torch.from_numpy(bboxes),
        'frame_resolution': torch.from_numpy(frame_resolutions)
    }


def extract_dataset_zip():
    dataset_path = 'dataset'
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)
    with ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(dataset_path)


def merge_train_with_test():
    dataset_path = 'dataset'
    train_variant = 'train'
    test_variant = 'test'
    train_test_variant = 'train_test'
    images_folder = 'images'
    labels_folder = 'labels'
    for calc_schema_variant in os.listdir(dataset_path):
        calc_schema_path = join(dataset_path, calc_schema_variant)
        train_dataset = join(calc_schema_path, train_variant)
        test_dataset = join(calc_schema_path, test_variant)
        train_test_dataset = join(calc_schema_path, train_test_variant)
        train_images = join(train_dataset, images_folder)
        train_labels = join(train_dataset, labels_folder)
        test_images = join(test_dataset, images_folder)
        test_labels = join(test_dataset, labels_folder)
        train_test_images = join(train_test_dataset, images_folder)
        train_test_labels = join(train_test_dataset, labels_folder)
        train_len = len(os.listdir(train_images))
        test_len = len(os.listdir(test_images))
        os.makedirs(train_test_dataset)
        copy_tree(train_dataset, train_test_dataset)
        for video_num in range(0, test_len):
            copy_tree(join(test_images, str(video_num)), join(train_test_images, str(video_num + train_len)))
            copy_tree(join(test_labels, str(video_num)), join(train_test_labels, str(video_num + train_len)))


if __name__ == '__main__':
    extract_dataset_zip()
    merge_train_with_test()
    print('The dataset has been unpacked')
