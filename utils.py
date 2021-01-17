from collections import OrderedDict
import warnings
from functools import partial
import pickle
import numpy as np

import torch


def save_checkpoint(model, epoch, checkpoint_path, remove_module_from_keys=False):
    state = {'state_dict': model.state_dict(), 'epoch': epoch + 1}

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict

    torch.save(state, checkpoint_path)
    print('Checkpoint saved to "{}"'.format(checkpoint_path))


def load_checkpoint(fpath):
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))

    return checkpoint['epoch']


def get_multiclass_label_map(label_map, N_CLASS):
    mc_label = np.zeros((N_CLASS, label_map.shape[0], label_map.shape[1]), dtype=np.bool)
    for i in range(1, N_CLASS + 1):
        mc_label[i - 1] = label_map == i
    return np.asarray(mc_label, dtype=np.float32)


def get_multiclass_label_map_cuda(label_map, N_CLASS):
    mc_label = torch.zeros((N_CLASS, label_map.shape[0], label_map.shape[1]), dtype=torch.uint8).cuda()
    for i in range(1, N_CLASS + 1):
        mc_label[i - 1] = torch.as_tensor(label_map == i, dtype=torch.uint8, device=torch.device('cuda'))
    return torch.as_tensor(mc_label, dtype=torch.float32, device=torch.device('cuda'))


def get_binary_label_map(label_map):
    bin_label = np.zeros((1, label_map.shape[0], label_map.shape[1]), dtype=np.bool)
    max_val = min(np.max(label_map), 3)
    for i in range(1, max_val + 1):
        bin_label[0, :, :] = np.logical_or(bin_label, label_map == i)
    if np.max(label_map) > 3:
        for i in range(4, np.max(label_map) + 1):
            bin_label[0, :, :] = bin_label[0, :, :] + ((label_map == i) * (i - 2))
    return np.asarray(bin_label, dtype=np.float32)


def multiclass_to_binary_label_map(mult_label_map, N_CLASS):
    bin_label = np.zeros((1, mult_label_map.shape[1], mult_label_map.shape[2]), dtype=np.float32)
    for i in range(0, N_CLASS):
        bin_label[0, :, :] = bin_label[0, :, :] + mult_label_map[i, :, :]
    return bin_label


def multiclass_to_binary_label_map_cuda(mult_label_map, N_CLASS):
    bin_label = torch.zeros((mult_label_map.shape[0], 1, mult_label_map.shape[2], mult_label_map.shape[3]),
                            dtype=torch.float32, device=torch.device('cuda'))
    for i in range(0, N_CLASS):
        bin_label[:, 0, :, :] = bin_label[:, 0, :, :] + mult_label_map[:, i, :, :]
    return bin_label


def dice_acc(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    return (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)


def jaccard_acc(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    return (intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection + smooth)


def intersect_acc(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    return (intersection + smooth) / (target.sum(dim=2).sum(dim=2) + smooth)


def soft_dice_loss(pred, target, smooth=1.):
    return torch.log(dice_acc(pred, target, smooth))


def soft_jaccard_loss(pred, target, smooth=1.):
    return torch.log(jaccard_acc(pred, target, smooth))