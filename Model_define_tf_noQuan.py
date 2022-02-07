import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

B=4

# CsiNet+(2019)
def Encoder(x, feedback_bits):
    with tf.compat.v1.variable_scope('Encoder'):
        # x = layers.Conv2D(128, 7, padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('sigmoid')(x)

        # x = layers.Conv2D(256, 7,padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)
        # # x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # x = layers.Conv2D(1, 7,padding='same')(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)

        # x = tf.reshape(x,(-1,126,8,16,2))
        # x = tf.transpose(x,(0,3,1,2,4))
        # x = tf.reshape(x,(-1,16,126*8*2))

        x = layers.Flatten()(x)
        # x = layers.GRU(units=512, return_sequences=True)(x)
        # x = layers.GRU(units=256, return_sequences=True)(x)
        # x = layers.Dense(units=1024, activation='relu')(x)
        # x = layers.Dense(units=512, activation='relu')(x)
        # x = layers.Dense(units=256, activation='relu')(x)
        x = layers.Dense(units=1024, activation='relu')(x)
        x = layers.Dense(units=int(feedback_bits // B), activation='sigmoid')(x)
        encoder_output = x
    return encoder_output


def Decoder(x,feedback_bits):
    # x = tf.reshape(x, (-1, 63, 64, 2))

    # x = layers.Conv2D(2, 7,padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # # x = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('sigmoid')(x)

    # x = layers.Conv2D(2, 7,padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU(alpha=0.1)(x)
    # # x = layers.Activation('relu')(x)
    # x = layers.UpSampling2D((2, 2))(x)

    # x = layers.Dense(units=256, activation='relu')(x)
    # x = layers.Dense(units=512, activation='relu')(x)
    # x = layers.Dense(units=1024, activation='relu')(x)
    # x = layers.GRU(units=256, return_sequences=True)(x)
    # x = layers.GRU(units=512, return_sequences=False)(x)
    
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dense(32256, activation='sigmoid')(x)
    x = layers.Reshape((126, 128, 2))(x)
    
    # x = layers.Conv2D(256, 7, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('sigmoid')(x)

    # x = layers.Conv2D(128, 7, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('sigmoid')(x)

    # x = layers.Conv2D(2, 7, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('sigmoid')(x)

    # for i in range(5):
    #     x = layers.Conv2D(8, 7, padding='SAME')(x_ini)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.LeakyReLU(alpha=0.1)(x)
    #     # x = layers.Activation('relu')(x)

    #     x = layers.Conv2D(16, 5, padding='SAME')(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.LeakyReLU(alpha=0.1)(x)
    #     # x = layers.Activation('relu')(x)

    #     x = layers.Conv2D(2, 3, padding='SAME')(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.Activation('tanh')(x)

    #     # x = layers.Dense(2, activation='linear')(x)

    #     x_ini = layers.Add()([x_ini, x])
    #     x_ini = layers.Activation('relu')(x_ini)
    decoder_output = x
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
