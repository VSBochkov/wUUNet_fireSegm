import numpy as np

from tensorflow.keras import backend as K


def get_multiclass_label_map(label_map, N_CLASS):
    mc_label = np.zeros((label_map.shape[0], label_map.shape[1], N_CLASS), dtype=np.bool)
    for i in range(1, N_CLASS + 1):
        mc_label[:, :, i - 1] = label_map == i
    return np.asarray(mc_label, dtype=np.float32)


def get_binary_label_map(label_map):
    bin_label = np.zeros((label_map.shape[0], label_map.shape[1], 1), dtype=np.bool)
    max_val = min(np.max(label_map), 3)
    for i in range(1, max_val + 1):
        bin_label[:, :, 0] = np.logical_or(bin_label, label_map == i)
    return np.asarray(bin_label, dtype=np.float32)


def multiclass_to_binary_label_map(mult_label_map, N_CLASS):
    bin_label = np.zeros((mult_label_map.shape[0], mult_label_map.shape[1], 1), dtype=np.float32)
    for i in range(0, N_CLASS):
        bin_label[:, :, 0] = bin_label[:, :, 0] + mult_label_map[:, :, i]
    return bin_label


def np_dice_acc(pred, target, smooth=1.):
    pred = np.ascontiguousarray(pred)
    target = np.ascontiguousarray(target)
    intersection = (pred * target).sum(axis=(1, 2))
    return (2. * intersection + smooth) / (pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2)) + smooth)


def np_jaccard_acc(pred, target, smooth=1.):
    pred = np.ascontiguousarray(pred)
    target = np.ascontiguousarray(target)
    intersection = (pred * target).sum(axis=(1, 2))
    return (intersection + smooth) / (pred.sum(axis=(1, 2)) + target.sum(axis=(1, 2)) - intersection + smooth)


def jaccard_acc(y_true, y_pred, smooth=1.):
    axes = [1, 2]
    intersection = K.sum(y_true * y_pred, axis=axes)
    sum_ = K.sum(y_true + y_pred, axis=axes)
    return (intersection + smooth) / (sum_ - intersection + smooth)


def bin_jaccard_acc(y_true, y_pred, smooth=1.):
    return jaccard_acc(K.max(y_true, axis=3, keepdims=True), y_pred, smooth)


def soft_jaccard_loss(y_true, y_pred):
    return -K.log(jaccard_acc(y_true, y_pred, 1.0))


def bce_jaccard_loss(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.mean(bce) + K.mean(soft_jaccard_loss(y_true, y_pred))


def bce_bin_jaccard_loss(y_true, y_pred):
    bin_y_true = K.max(y_true, axis=3, keepdims=True)
    bce = K.binary_crossentropy(bin_y_true, y_pred)
    return K.mean(bce) + K.mean(soft_jaccard_loss(bin_y_true, y_pred))
