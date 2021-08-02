import os.path

import keras.optimizers

from keras_unet_models import unet
from keras_utils import jaccard_acc, bce_jaccard_loss
from keras_dataset import Dataset, augm_train, augm_val

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

IS_LIGHT = False
dilation_rate = (1, 1)
COMB_TYPE = 'ow'
AUGMENTATION_ENABLED = True
INPUT_SIZE = 224
BS_TRAIN = 24
BS_VAL = BS_TRAIN * 2 - 4
LEARN_RATES = {
    0:  1e-3,
    50: 1e-4,
    100: 1e-5,
    150: 1e-6,
    200: 1e-7
}
#LEARN_RATES = {0: 1e-7}

BETAS = (0.9, 0.99)
WEIGHT_DECAY = 5e-4
EPOCH_N = 250
TOTAL_EPOCHS = 2000

DATASET_PATH = '../dataset/{}_{}'.format(COMB_TYPE, INPUT_SIZE)
MODEL_NAME = '{}_{}_{}'.format('unet_light' if IS_LIGHT else 'unet', INPUT_SIZE, COMB_TYPE)
OUTPUT_PATH = '../output/keras/{}'.format(MODEL_NAME)


if __name__ == '__main__':
    trainDataset = Dataset(os.path.join(DATASET_PATH, "train"), batch_size=BS_TRAIN, transforms=augm_train)

    bounds = []
    rates = []

    for iteration in range(int(TOTAL_EPOCHS / EPOCH_N)):
        for epoch_num in list(LEARN_RATES.keys()):
            bounds.append((iteration * EPOCH_N + epoch_num) * len(trainDataset))
        for learn_rate in LEARN_RATES.values():
            rates.append(learn_rate)
    bounds = bounds[1:]

    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=bounds,
        values=rates
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=BETAS[0],
        beta_2=BETAS[1],
        decay=WEIGHT_DECAY
    )

    model = unet(n_class=3, light=IS_LIGHT, dilation_rate=dilation_rate)
    model.compile(
        optimizer=optimizer,
        loss=bce_jaccard_loss,
        metrics=jaccard_acc
    )

    training_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_PATH + '/snapshots/best_val_loss_{epoch}.hdf5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_PATH + '/snapshots/best_val_acc_{epoch}.hdf5',
            monitor='val_jaccard_acc',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=OUTPUT_PATH + '/logdir')
    ]

    history = model.fit(
        trainDataset,
        validation_data=Dataset(os.path.join(DATASET_PATH, "test"), batch_size=BS_VAL, transforms=augm_val),
        epochs=TOTAL_EPOCHS,
        callbacks=training_callbacks
    )

    model.evaluate()

    print("FIT history = {}".format(history.history))