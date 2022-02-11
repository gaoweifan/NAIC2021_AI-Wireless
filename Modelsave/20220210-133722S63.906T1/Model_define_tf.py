from tkinter import X
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.

def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1, Num_.shape[1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)
    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


# CsiNet+(2019)
def Encoder(x, feedback_bits, trainable=True):
    B = 4
    with tf.compat.v1.variable_scope('Encoder'):
        # x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x_in = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        x = layers.Conv2D(64, 7, padding='same', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(16, 7, padding='same', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(32, 7, padding='same', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(16, 7, padding='same', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits // B), trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.Activation('sigmoid')(x)
        
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output


def Decoder(x,feedback_bits, trainable=True):
    B = 4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.reshape(decoder_input, (-1, int(feedback_bits//B)))
    x = layers.Dense(32256, trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.Activation('sigmoid')(x)
    x = layers.Reshape((126, 128, 2))(x)

    x = layers.Conv2D(32, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('relu')(x)

    x = layers.Conv2D(16, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('relu')(x)

    x = layers.Conv2D(8, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('relu')(x)

    x = layers.Conv2D(16, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('relu')(x)

    x = layers.Conv2D(8, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('relu')(x)

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    # x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Activation('sigmoid')(x)
    
    decoder_output = x
    return decoder_output

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x_ini = layers.Activation('sigmoid')(x)

    for i in range(3):
        x = layers.Conv2D(8, 7, padding='SAME', trainable=trainable)(x_ini)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(16, 5, padding='SAME', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        # x = layers.Activation('relu')(x)

        x = layers.Conv2D(2, 3, padding='SAME', trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.Activation('tanh')(x)

        # x = layers.Dense(2, activation='linear', trainable=trainable)(x)

        x_ini = layers.Add()([x_ini, x])
        x_ini = layers.Activation('relu')(x_ini)
    decoder_output = x_ini
    return decoder_output



def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}
