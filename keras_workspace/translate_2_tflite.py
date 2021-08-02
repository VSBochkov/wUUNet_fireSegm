import os

import tensorflow as tf

from keras_unet_models import unet, wuunet

OUTPUT_DIR = '../output/translate'

# MODEL_NAME = 'unet_224_n3_jaccard'
# MODEL_NAME = 'unet_224_n3_jaccard_dilated'
# MODEL_NAME = 'unet_224_n3_jaccard_light'
# MODEL_NAME = 'unet_224_n3_jaccard_light_dilated'
# MODEL_NAME = 'wunet_224_n3_jaccard'
# MODEL_NAME = 'wunet_224_n3_jaccard_dilated'
# MODEL_NAME = 'wunet_224_n3_jaccard_light'
MODEL_NAME = 'wunet_224_n3_jaccard_light_dilated'

EXPORTED_TF_MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_NAME)

# model = unet(n_class=3, batch_size=1, light=False, dilation_rate=(1, 1))
# model = unet(n_class=3, batch_size=1, light=False, dilation_rate=(2, 2))
# model = unet(n_class=3, batch_size=1, light=True, dilation_rate=(1, 1))
# model = unet(n_class=3, batch_size=1, light=True, dilation_rate=(2, 2))
# model = wuunet(n_class=3, batch_size=1, light=False, dilation_rate=(1, 1))
# model = wuunet(n_class=3, batch_size=1, light=False, dilation_rate=(2, 2))
# model = wuunet(n_class=3, batch_size=1, light=True, dilation_rate=(1, 1))
model = wuunet(n_class=3, batch_size=1, light=True, dilation_rate=(2, 2))

# model.load_weights('../output/keras/unet_224_ow/snapshots/best_val_acc_203.hdf5')
# model.load_weights('../output/keras/unet_light_224_ow/snapshots/best_val_acc_103.hdf5')
# model.load_weights('../output/keras/wuunet_224_ow/snapshots/best_val_mult_acc_215.hdf5')
# model.load_weights('../output/keras/wuunet_light_224_ow/snapshots/best_val_mult_acc_88.hdf5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model
with open(os.path.join(OUTPUT_DIR, MODEL_NAME + ".tflite"), 'wb') as f:
    f.write(tflite_model)
