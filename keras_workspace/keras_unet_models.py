import keras.layers
from keras.models import *
from keras.layers import *

import tensorflow as tf


class SegmentMaskBinarize(keras.layers.Layer):
    def __init__(self, input_dim, threshold):
        super(SegmentMaskBinarize, self).__init__()
        self.threshold = threshold
        self.total = tf.Variable(initial_value=tf.zeros(input_dim, input_dim), trainable=False)

    def call(self, inputs, **kwargs):
        thresholded = tf.subtract(inputs, self.threshold)
        clipped = tf.clip_by_value(thresholded, clip_value_min=0.0, clip_value_max=1.0)
        argmax = tf.argmax(clipped)
        max = tf.reduce_max(clipped)
        self.total.assign_add(argmax + tf.cast(tf.greater(max, 0), tf.int64))
        return self.total


def double_conv(inputs, out_channels, dilation_rate):
    kernel_size = 3
    c1 = Conv2D(out_channels, kernel_size, activation='relu', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(inputs)
    return Conv2D(out_channels, kernel_size, activation='relu', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(c1)


def unet(n_class=3, batch_size=None, light=False, dilation_rate=(1, 1)):
    width = 32 if light else 64
    updown_sampling_size = (2, 2)
    input_shape = (224, 224, 3)
    upsample_interp = 'bilinear'
    inputs = Input(input_shape, batch_size=batch_size)
    dconv_down1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1)
    dconv_down2 = double_conv(pool1, width * 2, dilation_rate=dilation_rate)
    pool2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2)
    dconv_down3 = double_conv(pool2, width * 4, dilation_rate=dilation_rate)
    pool3 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3)
    bottleneck = double_conv(pool3, width * 8, dilation_rate=dilation_rate)
    upsampled3 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck)
    merge3 = concatenate([upsampled3, dconv_down3], axis=3)
    dconv_up3 = double_conv(merge3, width * 4, dilation_rate=dilation_rate)
    upsampled2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3)
    merge2 = concatenate([upsampled2, dconv_down2], axis=3)
    dconv_up2 = double_conv(merge2, width * 2, dilation_rate=dilation_rate)
    upsampled1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2)
    merge1 = concatenate([upsampled1, dconv_down1], axis=3)
    dconv_up1 = double_conv(merge1, width, dilation_rate=dilation_rate)
    conv_last = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(dconv_up1)

    return Model(inputs=inputs, outputs=conv_last)


def unet_infer(threshold, n_class=3, batch_size=None, light=False, dilation_rate=(1, 1)):
    unet_inference = Sequential()
    unet_inference.add(unet(n_class=n_class, batch_size=batch_size, light=light, dilation_rate=dilation_rate))
    unet_inference.add(SegmentMaskBinarize(input_dim=224, threshold=threshold))
    return unet_inference


def wuunet(n_class=3, batch_size=None, light=False, dilation_rate=(1, 1)):
    width = 32 if light else 64
    updown_sampling_size = (2, 2)
    input_shape = (224, 224, 3)
    upsample_interp = 'bilinear'

    inputs = Input(input_shape, batch_size=batch_size)

    dconv_down1_1 = double_conv(inputs, width, dilation_rate=dilation_rate)
    pool1_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1_1)
    dconv_down2_1 = double_conv(pool1_1, width * 2, dilation_rate=dilation_rate)
    pool2_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2_1)
    dconv_down3_1 = double_conv(pool2_1, width * 4, dilation_rate=dilation_rate)
    pool3_1 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3_1)
    bottleneck_1 = double_conv(pool3_1, width * 8, dilation_rate=dilation_rate)
    upsampled3_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck_1)
    merge3_1 = concatenate([upsampled3_1, dconv_down3_1], axis=3)
    dconv_up3_1 = double_conv(merge3_1, width * 4, dilation_rate=dilation_rate)
    upsampled2_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3_1)
    merge2_1 = concatenate([upsampled2_1, dconv_down2_1], axis=3)
    dconv_up2_1 = double_conv(merge2_1, width * 2, dilation_rate=dilation_rate)
    upsampled1_1 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2_1)
    merge1_1 = concatenate([upsampled1_1, dconv_down1_1], axis=3)
    dconv_up1_1 = double_conv(merge1_1, width, dilation_rate=dilation_rate)
    out_bin = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name='out_bin')(dconv_up1_1)

    inputs_bin = concatenate([inputs, out_bin], axis=3)

    dconv_down1_2 = double_conv(inputs_bin, width, dilation_rate=dilation_rate)
    pool1_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down1_2)
    merge_enc1_2 = concatenate([pool1_2, dconv_down2_1, dconv_up2_1], axis=3)
    dconv_down2_2 = double_conv(merge_enc1_2, width * 2, dilation_rate=dilation_rate)
    pool2_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down2_2)
    merge_enc2_2 = concatenate([pool2_2, dconv_down3_1, dconv_up3_1], axis=3)
    dconv_down3_2 = double_conv(merge_enc2_2, width * 4, dilation_rate=dilation_rate)
    pool3_2 = MaxPooling2D(pool_size=updown_sampling_size)(dconv_down3_2)
    merge_enc3_2 = concatenate([pool3_2, bottleneck_1], axis=3)
    bottleneck_2 = double_conv(merge_enc3_2, width * 8, dilation_rate=dilation_rate)
    upsampled3_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(bottleneck_2)
    merge_dec3_2 = concatenate([upsampled3_2, dconv_down3_1, dconv_up3_1, dconv_down3_2])
    dconv_up3_2 = double_conv(merge_dec3_2, width * 4, dilation_rate=dilation_rate)
    upsampled2_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up3_2)
    merge_dec2_2 = concatenate([upsampled2_2, dconv_down2_1, dconv_up2_1, dconv_down2_2])
    dconv_up2_2 = double_conv(merge_dec2_2, width * 2, dilation_rate=dilation_rate)
    upsampled1_2 = UpSampling2D(size=updown_sampling_size, interpolation=upsample_interp)(dconv_up2_2)
    merge_dec1_2 = concatenate([upsampled1_2, dconv_down1_1, inputs_bin], axis=3)
    dconv_up1_2 = double_conv(merge_dec1_2, width, dilation_rate=dilation_rate)
    out_mult = Conv2D(n_class, kernel_size=1, activation='sigmoid', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name='out_mult')(dconv_up1_2)

    return Model(
        inputs=inputs,
        outputs={
            'out_bin': out_bin,
            'out_mult': out_mult
        }
    )


def wuunet_infer(threshold, n_class=3, batch_size=None, light=False, dilation_rate=(1, 1)):
    wuunet_model = wuunet(n_class=n_class, batch_size=batch_size, light=light, dilation_rate=dilation_rate)
    out_binarized = SegmentMaskBinarize(input_dim=224, threshold=threshold)(wuunet_model.output['out_mult'])
    return Model(inputs=wuunet_model.inputs, outputs=out_binarized)