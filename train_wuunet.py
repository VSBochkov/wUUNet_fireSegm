import json
import os
from collections import defaultdict
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import MulticlassFiresegmDataset, augm_train, augm_val
from unet_models import wuunet, uunet
from utils import (
    soft_dice_loss, soft_jaccard_loss, dice_acc, jaccard_acc, load_pretrained_weights, save_checkpoint,
    multiclass_to_binary_label_map_cuda
)

N_CLASS = 3  # 1
MAIN_METRIC = 'jaccard'     # 'dice'
FULL_SIZE = True
WIDE = True
AUGMENTATION_ENABLED = True
INPUT_SIZE = (224, 224)
BS_TRAIN = 12
BS_VAL = BS_TRAIN * 2
LEARN_RATES = {
    0:  1e-3,
    50: 1e-4,
    100: 1e-5,
    150: 1e-6,
    240: 1e-7
}
#LEARN_RATES = {
#    0: 1e-7
#}

WEIGHT_DECAY = 5e-4
EPOCH_N = 240
TRAIN_GOLDGOAL_PERIOD = int(EPOCH_N / 4)
EVAL_EVERY = 1


best_mult_acc_checkpoint_filename = 'mult_acc.pth.tar'
best_bin_acc_checkpoint_filename = 'bin_acc.pth.tar'
best_mult_trainval_acc_checkpoint_filename = 'mult_trainval_acc.pth.tar'
best_bin_trainval_acc_checkpoint_filename = 'bin_trainval_acc.pth.tar'
best_loss_checkpoint_filename = 'loss.pth.tar'
best_mult_jaccard_loss_checkpoint_filename = 'mult_loss.pth.tar'
best_mult_bce_loss_checkpoint_filename = 'mult_bce_loss.pth.tar'
last_checkpoint_filename = 'last.pth.tar'


def get_checkpoint_filename(epoch_num):
    return '{}.pth.tar'.format(epoch_num)


def calc_loss(pred_bin, pred_mult, target, metrics_accumulated):
    target_bin = multiclass_to_binary_label_map_cuda(target, N_CLASS)
    bce_bin = F.binary_cross_entropy_with_logits(pred_bin, target_bin)
    pred_bin = torch.sigmoid(pred_bin)
    bce_mult = F.binary_cross_entropy_with_logits(pred_mult, target)
    pred_mult = torch.sigmoid(pred_mult)
    main_bin_metric_loss, main_bin_metric_acc = 0., 0.
    main_mult_metric_loss, main_mult_metric_acc = 0., 0.
    if MAIN_METRIC is 'dice':
        main_bin_metric_loss = -1 * (soft_dice_loss(pred_bin, target_bin).mean())
        main_bin_metric_acc = dice_acc(pred_bin, target_bin).mean()
        main_mult_metric_loss = -1 * (soft_dice_loss(pred_mult, target).mean())
        main_mult_metric_acc = dice_acc(pred_mult, target).mean()
    elif MAIN_METRIC is 'jaccard':
        main_bin_metric_loss = -1 * (soft_jaccard_loss(pred_bin, target_bin).mean())
        main_bin_metric_acc = jaccard_acc(pred_bin, target_bin).mean()
        main_mult_metric_loss = -1 * (soft_jaccard_loss(pred_mult, target).mean())
        main_mult_metric_acc = jaccard_acc(pred_mult, target).mean()
    loss = bce_mult + bce_bin + main_bin_metric_loss + main_mult_metric_loss
    metrics_accumulated['bce_bin'] += bce_bin.data.cpu().numpy() * target.size(0)
    metrics_accumulated['bce_mult'] += bce_mult.data.cpu().numpy() * target.size(0)
    metrics_accumulated[MAIN_METRIC + '_bin_loss'] += main_bin_metric_loss.data.cpu().numpy() * target.size(0)
    metrics_accumulated[MAIN_METRIC + '_bin_acc'] += main_bin_metric_acc.data.cpu().numpy() * target.size(0)
    metrics_accumulated[MAIN_METRIC + '_mult_loss'] += main_mult_metric_loss.data.cpu().numpy() * target.size(0)
    metrics_accumulated[MAIN_METRIC + '_mult_acc'] += main_mult_metric_acc.data.cpu().numpy() * target.size(0)
    metrics_accumulated['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


def get_4dig(a):
    return "{0:.4f}".format(a)


def log_epoch(stage, metrics_accumulated, epoch_samples, epoch_num):
    logger.add_scalar('bce_bin_{}_epoch'.format(stage), metrics_accumulated['bce_bin'] / epoch_samples, epoch_num)
    logger.add_scalar('bce_mult_{}_epoch'.format(stage), metrics_accumulated['bce_mult'] / epoch_samples, epoch_num)
    logger.add_scalar('loss_{}_epoch'.format(stage), metrics_accumulated['loss'] / epoch_samples,  epoch_num)
    logger.add_scalar('{}_bin_acc_{}_epoch'.format(MAIN_METRIC, stage),
                      metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples,
                      epoch_num)
    logger.add_scalar('{}_mult_acc_{}_epoch'.format(MAIN_METRIC, stage),
                      metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples,
                      epoch_num)

    loss_aver, bce_bin_aver, bce_mult_aver, main_bin_metric_loss_aver, main_bin_metric_acc_aver, main_mult_metric_loss_aver, main_mult_metric_acc_aver = \
        metrics_accumulated['loss'] / epoch_samples, \
        metrics_accumulated['bce_bin'] / epoch_samples, \
        metrics_accumulated['bce_mult'] / epoch_samples, \
        metrics_accumulated[MAIN_METRIC + '_bin_loss'] / epoch_samples, \
        metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples, \
        metrics_accumulated[MAIN_METRIC + '_mult_loss'] / epoch_samples, \
        metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples

    print('[#{}, {}] Loss {}'.format(epoch_num, stage, get_4dig(loss_aver)))
    print('BIN: BCE {} {}: acc {}, loss {}'.format(
        get_4dig(bce_bin_aver), MAIN_METRIC, get_4dig(main_bin_metric_acc_aver), get_4dig(main_bin_metric_loss_aver)
    ))
    print('MUL: BCE {} {}: acc {}, loss {}'.format(
        get_4dig(bce_mult_aver), MAIN_METRIC, get_4dig(main_mult_metric_acc_aver), get_4dig(main_mult_metric_loss_aver)
    ))


def save_train_results_json():
    global best_loss, best_bin_acc, best_mult_acc, best_loss_epoch, best_bin_acc_epoch, best_mult_acc_epoch, \
        best_bin_trainval_acc, best_mult_trainval_acc, best_bin_trainval_acc_epoch, best_mult_trainval_acc_epoch, \
        train_bin_acc, train_mult_acc, best_mult_bce_loss, best_mult_bce_loss_epoch, best_mult_jaccard_loss, \
        best_mult_jaccard_loss_epoch, epoch_offset
    json.dump({
        'best_bin_acc': {
            'epoch_num': best_bin_acc_epoch,
            'value': best_bin_acc
        },
        'best_mult_acc': {
            'epoch_num': best_mult_acc_epoch,
            'value': best_mult_acc
        },
        'best_bin_trainval_acc': {
            'epoch_num': best_bin_trainval_acc_epoch,
            'value': best_bin_trainval_acc
        },
        'best_mult_trainval_acc': {
            'epoch_num': best_mult_trainval_acc_epoch,
            'value': best_mult_trainval_acc
        },
        'best_loss': {
            'epoch_num': best_loss_epoch,
            'value': best_loss
        },
        'best_mult_bce_loss': {
            'epoch_num': best_mult_bce_loss_epoch,
            'value': best_mult_bce_loss
        },
        'best_mult_jaccard_loss': {
            'epoch_num': best_mult_jaccard_loss_epoch,
            'value': best_mult_jaccard_loss
        },
        'last_epoch_num': epoch,
        'epoch_offset': epoch_offset
    }, open(join(snapshot_dir, 'train_results.json'), 'w'))


def finish():
    global epoch_offset, best_mult_acc_epoch, best_mult_trainval_acc_epoch, best_bin_trainval_acc_epoch, best_bin_acc_epoch, best_loss_epoch
    f = open(join(snapshot_dir, 'finish'), 'w')
    f.write(str(epoch))
    f.close()
    if epoch >= epoch_offset + EPOCH_N + TRAIN_GOLDGOAL_PERIOD:
        f = open(join(snapshot_dir, 'the_end'), 'w')
        f.write(str(epoch))
        f.close()
    epoch_offset = max([
        best_mult_trainval_acc_epoch,
        best_mult_acc_epoch,
        best_bin_trainval_acc_epoch,
        best_bin_acc_epoch,
        best_loss_epoch])
    f = open(join(snapshot_dir, 'save_all_results'), 'w')
    f.write(str(1))
    f.close()
    save_train_results_json()


train_bin_acc = 0.
train_mult_acc = 0.


def train(epoch):
    global train_bin_acc, train_mult_acc
    model.train()
    epoch_samples = 0
    metrics_accumulated = defaultdict(float)
    for batch_idx, data in enumerate(tqdm(dataloader_train)):
        optimizer.zero_grad()
        out_bin, out_mult = model(data['imgs'].cuda())
        loss = calc_loss(out_bin.cuda(), out_mult.cuda(), data['lbls'].cuda(), metrics_accumulated)
        loss.backward()
        optimizer.step()
        epoch_samples += len(data['lbls'])
    log_epoch('train', metrics_accumulated, epoch_samples, epoch)
    train_bin_acc = metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples
    train_mult_acc = metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples
    torch.cuda.empty_cache()


def eval(epoch):
    model.eval()
    global best_loss, best_bin_acc, best_mult_acc, best_loss_epoch, best_bin_acc_epoch, best_mult_acc_epoch, \
        best_bin_trainval_acc, best_mult_trainval_acc, best_bin_trainval_acc_epoch, best_mult_trainval_acc_epoch, \
        best_mult_jaccard_loss, best_mult_jaccard_loss_epoch, best_mult_bce_loss, best_mult_bce_loss_epoch, \
        train_bin_acc, train_mult_acc, best_mult_trainval_acc_checkpoint_filename, \
        best_bin_trainval_acc_checkpoint_filename, best_bin_acc_checkpoint_filename, \
        best_mult_acc_checkpoint_filename, best_loss_checkpoint_filename, \
        best_mult_jaccard_loss_checkpoint_filename, best_mult_bce_loss_checkpoint_filename
    
    epoch_samples = 0
    metrics_accumulated = defaultdict(float)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader_val)):
            outputs_bin, outputs_mul = model(data['imgs'].cuda())
            calc_loss(outputs_bin.cuda(), outputs_mul.cuda(), data['lbls'].cuda(), metrics_accumulated)
            epoch_samples += len(data['lbls'])

    log_epoch('eval', metrics_accumulated, epoch_samples, epoch)
    saved = False
    if metrics_accumulated['loss'] / epoch_samples < best_loss:
        best_loss = metrics_accumulated['loss'] / epoch_samples
        best_loss_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_loss_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    if metrics_accumulated[MAIN_METRIC + '_mult_loss'] / epoch_samples < best_mult_jaccard_loss:
        best_mult_jaccard_loss = metrics_accumulated[MAIN_METRIC + '_mult_loss'] / epoch_samples
        best_mult_jaccard_loss_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_mult_jaccard_loss_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    if metrics_accumulated['bce_mult'] / epoch_samples < best_mult_bce_loss:
        best_mult_bce_loss = metrics_accumulated['bce_mult'] / epoch_samples
        best_mult_bce_loss_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_mult_bce_loss_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    if metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples > best_bin_acc:
        best_bin_acc = metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples
        best_bin_acc_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_bin_acc_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    if metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples > best_mult_acc:
        best_mult_acc = metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples
        best_mult_acc_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_mult_acc_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    if train_bin_acc + metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples > best_bin_trainval_acc:
        best_bin_trainval_acc = train_bin_acc + metrics_accumulated[MAIN_METRIC + '_bin_acc'] / epoch_samples
        best_bin_trainval_acc_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_bin_trainval_acc_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    if train_mult_acc + metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples > best_mult_trainval_acc:
        best_mult_trainval_acc = train_mult_acc + metrics_accumulated[MAIN_METRIC + '_mult_acc'] / epoch_samples
        best_mult_trainval_acc_epoch = epoch
        checkpoint_path = join(snapshot_dir, best_mult_trainval_acc_checkpoint_filename)
        save_checkpoint(model, epoch, checkpoint_path)
        saved = True
    checkpoint_path = join(snapshot_dir, last_checkpoint_filename)
    save_checkpoint(model, epoch, checkpoint_path)
    if saved and os.path.exists(join(snapshot_dir, 'save_all_results')):
        checkpoint_path = join(snapshot_dir, get_checkpoint_filename(epoch))
        save_checkpoint(model, epoch, checkpoint_path)
    save_train_results_json()
    torch.cuda.empty_cache()


def recursive_printout_model(model):
    print(model)
    for name, module in model._modules.items():
        recursive_printout_model(module)


if __name__ == '__main__':
    dataset_train = MulticlassFiresegmDataset(
        'train', augm_train if AUGMENTATION_ENABLED else None, input_size=INPUT_SIZE, full_size=FULL_SIZE)
    dataset_val = MulticlassFiresegmDataset(
        'test', augm_val if AUGMENTATION_ENABLED else None, input_size=INPUT_SIZE, full_size=FULL_SIZE)

    print('dataset_train len = {}'.format(len(dataset_train)))
    print('dataset_val len = {}'.format(len(dataset_val)))

    dataloader_train = dataset_train.get_dataloader(batch_size=BS_TRAIN, shuffle=True, num_workers=12)
    dataloader_val = dataset_val.get_dataloader(batch_size=BS_VAL, shuffle=True, num_workers=12)

    model = wuunet(N_CLASS).cuda() if WIDE else uunet(N_CLASS).cuda()
    if FULL_SIZE:
        model_name = '{}_{}_n{}_fs_{}'
    else:
        model_name = '{}_{}_n{}_ow_{}'
    model_name = model_name.format(model.__class__.__name__, INPUT_SIZE[0], N_CLASS, MAIN_METRIC)
    snapshot_dir = 'output/{}/snapshots'.format(model_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_file_path = join(snapshot_dir, last_checkpoint_filename)
    logger = SummaryWriter('output/{}/logdir'.format(model_name))

    lr_epoch_num = 0
    best_bin_acc = -np.inf
    best_bin_acc_epoch = 0
    best_mult_acc = -np.inf
    best_mult_acc_epoch = 0
    best_mult_trainval_acc = -np.inf
    best_mult_trainval_acc_epoch = 0
    best_bin_trainval_acc = -np.inf
    best_bin_trainval_acc_epoch = 0
    best_loss = np.inf
    best_loss_epoch = 0
    best_mult_jaccard_loss = np.inf
    best_mult_jaccard_loss_epoch = 0
    best_mult_bce_loss = np.inf
    best_mult_bce_loss_epoch = 0
    if os.path.exists(snapshot_file_path):
        train_results = json.load(open(join(snapshot_dir, 'train_results.json'), 'r'))
        best_mult_acc = float(train_results['best_mult_acc']['value'])
        best_mult_acc_epoch = int(train_results['best_mult_acc']['epoch_num'])
        best_mult_trainval_acc = float(train_results['best_mult_trainval_acc']['value'])
        best_mult_trainval_acc_epoch = int(train_results['best_mult_trainval_acc']['epoch_num'])
        best_bin_acc = float(train_results['best_bin_acc']['value'])
        best_bin_acc_epoch = int(train_results['best_bin_acc']['epoch_num'])
        best_bin_trainval_acc = float(train_results['best_bin_trainval_acc']['value'])
        best_bin_trainval_acc_epoch = int(train_results['best_bin_trainval_acc']['epoch_num'])
        best_loss = float(train_results['best_loss']['value'])
        best_loss_epoch = int(train_results['best_loss']['epoch_num'])
        best_mult_jaccard_loss = float(train_results['best_mult_jaccard_loss']['value'])
        best_mult_jaccard_loss_epoch = int(train_results['best_mult_jaccard_loss']['epoch_num'])
        best_mult_bce_loss = float(train_results['best_mult_bce_loss']['value'])
        best_mult_bce_loss_epoch = int(train_results['best_mult_bce_loss']['epoch_num'])
        epoch_offset = int(train_results['epoch_offset'])
        if os.path.exists(join(snapshot_dir, 'finish')):
            checkpoints_dict = {
                best_mult_acc_epoch: best_mult_acc_checkpoint_filename,
                best_bin_acc_epoch: best_bin_acc_checkpoint_filename,
                best_mult_trainval_acc_epoch: best_mult_trainval_acc_checkpoint_filename,
                best_bin_trainval_acc_epoch: best_bin_trainval_acc_checkpoint_filename,
                best_loss_epoch: best_loss_checkpoint_filename
            }
            snapshot_file_path = join(snapshot_dir, checkpoints_dict[epoch_offset])
            os.remove(join(snapshot_dir, 'finish'))
        LEARN_RATES = {epoch_num + epoch_offset: LEARN_RATES[epoch_num] for epoch_num in LEARN_RATES}
        EPOCH_N = EPOCH_N + epoch_offset
        checkpoint_epoch = load_pretrained_weights(model, snapshot_file_path)
        prev_epoch_num = 0
        for lr_epoch in LEARN_RATES:
            if lr_epoch > checkpoint_epoch:
                break
            prev_epoch_num = lr_epoch
        lr_epoch_num = prev_epoch_num
    else:
        checkpoint_epoch = 0
        epoch_offset = 0

    print('start training {} model'.format(model.__class__.__name__))
    print('\tcheckpoint epoch = {}'.format(checkpoint_epoch))
    print('\ttraining finish epoch num = {} (+{} GOLDGOAL)'.format(EPOCH_N, TRAIN_GOLDGOAL_PERIOD))
    print('\tLEARNING RATES MAP = {}'.format(LEARN_RATES))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATES[lr_epoch_num], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))
    print(optimizer)

    torch.cuda.empty_cache()

    for epoch in tqdm(range(checkpoint_epoch, EPOCH_N)):
        if epoch in LEARN_RATES:
            for params in optimizer.param_groups:
                params['lr'] = LEARN_RATES[epoch]
            print(optimizer)

        print('=> Training epoch {}'.format(epoch))
        train(epoch)

        if (epoch + 1) % EVAL_EVERY == 0:
            print('=> Validation epoch {}'.format(epoch))
            eval(epoch)

        print('best_bin_acc:\t\tepoch #{}, value = {}'.format(best_bin_acc_epoch, get_4dig(best_bin_acc)))
        print('best_mult_acc:\t\tepoch #{}, value = {}'.format(best_mult_acc_epoch, get_4dig(best_mult_acc)))
        print('best_bin_trainval_acc:\tepoch #{}, value = {}'.format(
            best_bin_trainval_acc_epoch, get_4dig(best_bin_trainval_acc)))
        print('best_mult_trainval_acc:\tepoch #{}, value = {}'.format(
            best_mult_trainval_acc_epoch, get_4dig(best_mult_trainval_acc)))
        print('best_loss:\t\tepoch #{}, value = {}'.format(best_loss_epoch, get_4dig(best_loss)))
        print('best_mult_bce_loss:\tepoch #{}, value = {}'.format(best_mult_bce_loss_epoch, get_4dig(best_mult_bce_loss)))
        print('best_mult_jaccard_loss:\tepoch #{}, value = {}'.format(best_mult_jaccard_loss_epoch, get_4dig(best_mult_jaccard_loss)))
        print()
    if checkpoint_epoch < EPOCH_N - 1:
        checkpoint_epoch = EPOCH_N - 1
    if checkpoint_epoch - best_mult_trainval_acc_epoch > 10:
        epoch = checkpoint_epoch - 1
        while best_mult_trainval_acc_epoch != epoch and epoch < EPOCH_N + TRAIN_GOLDGOAL_PERIOD:
            epoch += 1
            if epoch in LEARN_RATES:
                for params in optimizer.param_groups:
                    params['lr'] = LEARN_RATES[epoch]
                print(optimizer)

            print('=> Training epoch {}'.format(epoch))
            train(epoch)

            if (epoch + 1) % EVAL_EVERY == 0:
                print('=> Validation epoch {}'.format(epoch))
                eval(epoch)

            print('best_bin_acc:\t\tepoch #{}, value = {}'.format(best_bin_acc_epoch, get_4dig(best_bin_acc)))
            print('best_mult_acc:\t\tepoch #{}, value = {}'.format(best_mult_acc_epoch, get_4dig(best_mult_acc)))
            print('best_bin_trainval_acc:\tepoch #{}, value = {}'.format(
                best_bin_trainval_acc_epoch, get_4dig(best_bin_trainval_acc)))
            print('best_mult_trainval_acc:\tepoch #{}, value = {}'.format(
                best_mult_trainval_acc_epoch, get_4dig(best_mult_trainval_acc)))
            print('best_loss:\t\tepoch #{}, value = {}'.format(best_loss_epoch, get_4dig(best_loss)))
            print('best_mult_bce_loss:\tepoch #{}, value = {}'.format(best_mult_bce_loss_epoch, get_4dig(best_mult_bce_loss)))
            print('best_mult_jaccard_loss:\tepoch #{}, value = {}'.format(best_mult_jaccard_loss_epoch, get_4dig(best_mult_jaccard_loss)))
            print()
    finish()
