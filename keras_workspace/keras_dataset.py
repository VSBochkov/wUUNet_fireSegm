import math

import os
import cv2
import numpy as np
from keras.utils import Sequence
from keras_utils import get_multiclass_label_map

from albumentations import (
    Blur, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
    RandomBrightnessContrast, IAASharpen, IAAEmboss, OneOf, Compose
)


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


# augm_train = _augm(0.75)
# augm_val = _augm(0.5)

augm_train = _augm(1)
augm_val = _augm(1)


class Dataset(Sequence):
    def __init__(self, database_path, batch_size=48, transforms=None, shuffle=True):
        super(Dataset, self).__init__()
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle = shuffle
        self.copies = 1  # 3 if transforms is not None else 1
        self.indexes, self.data_map, self.label_map = self.get_images_and_labels(database_path)

    @staticmethod
    def hash_key(video_num, sample_num, copy_num):
        return video_num * 100_00 + sample_num * 100 + copy_num

    def get_images_and_labels(self, database_path):
        indexes = []
        data_map = {}
        label_map = {}
        images_dir = os.path.join(database_path, 'images')
        labels_dir = os.path.join(database_path, 'labels')
        for video_num in range(0, len(os.listdir(images_dir))):
            img_video_path = os.path.join(images_dir, str(video_num))
            lbl_video_path = os.path.join(labels_dir, str(video_num))
            for sample_num in range(0, len(os.listdir(img_video_path))):
                image = cv2.imread(os.path.join(img_video_path, str(sample_num) + '.jpg'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = np.load(os.path.join(lbl_video_path, str(sample_num) + '.npy'))
                one_hot_label = get_multiclass_label_map(label, 3)
                for copy_num in range(0, self.copies):
                    indexes.append(Dataset.hash_key(video_num, sample_num, copy_num))
                    data_map[Dataset.hash_key(video_num, sample_num, copy_num)] = image
                    label_map[Dataset.hash_key(video_num, sample_num, copy_num)] = one_hot_label
        return np.array(indexes), data_map, label_map

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.indexes[idx * self.batch_size: min((idx + 1) * self.batch_size, len(self.indexes))]
        images, labels = [], []
        for key in batch:
            image = self.data_map[key].copy()
            label = self.label_map[key]
            if self.transforms is not None:
                image = self.transforms(image=image)['image']
            images.append(np.asarray(image, np.float32) / 255.0)
            labels.append(label)
        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
