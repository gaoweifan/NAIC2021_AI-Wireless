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

image_size = 128  # We'll resize input images to this size
patch_size = 32  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 128
DEC_PROJECTION_DIM = 64
LAYER_NORM_EPS = 1e-6
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
dec_layers = 8
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]
mlp_head_units = [2048, 1024, 512]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate, trainable=True):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu, trainable=trainable)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, trainable):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim,trainable=trainable)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim,trainable=trainable
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

'''
# transformer
def Encoder(x, feedback_bits, trainable=True):
    B = 4
    with tf.compat.v1.variable_scope('Encoder'):
        # pad 0.5 to 128*128
        x=x-0.5
        x=layers.ZeroPadding2D(padding=(1, 0))(x)
        x=x+0.5

        x = layers.Conv2D(32, 7, padding='same', trainable=trainable,name="enc_conv_1")(x)
        x = layers.BatchNormalization(trainable=trainable,name="enc_bn_1")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(16, 7, padding='same', trainable=trainable,name="enc_conv_2")(x)
        x = layers.BatchNormalization(trainable=trainable,name="enc_bn_2")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(2, 7, padding='same', trainable=trainable,name="enc_conv_3")(x)
        x = layers.BatchNormalization(trainable=trainable,name="enc_bn_3")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        x = layers.Conv2D(32, 7, padding='same', trainable=trainable,name="enc_conv_4")(x)
        x = layers.BatchNormalization(trainable=trainable,name="enc_bn_4")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(16, 7, padding='same', trainable=trainable,name="enc_conv_5")(x)
        x = layers.BatchNormalization(trainable=trainable,name="enc_bn_5")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(2, 7, padding='same', trainable=trainable,name="enc_conv_6")(x)
        x = layers.BatchNormalization(trainable=trainable,name="enc_bn_6")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        # Augment data.
        # augmented = data_augmentation(inputs)
        # x = layers.Normalization()(x)
        # Create patches.
        patches = Patches(patch_size)(x)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim, trainable=trainable)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS, trainable=trainable)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1, trainable=trainable
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS, trainable=trainable)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1, trainable=trainable)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS,trainable=trainable,name="enc_ln_3")(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.1)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1,trainable=trainable)
        # Classify outputs.
        x = layers.Dense(units=int(feedback_bits // B), activation='sigmoid',trainable=trainable,name="enc_dense_5")(features)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output

def Decoder(x,feedback_bits, trainable=True):
    B = 4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.reshape(decoder_input, (-1, int(feedback_bits//B)))

    x = layers.Dense(num_patches*projection_dim, trainable=trainable,name="dec_dense_1")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_1")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    inputs = layers.Reshape((num_patches, projection_dim))(x)
    x = layers.Dense(DEC_PROJECTION_DIM,trainable=trainable,name="dec_dense_2")(inputs)

    # Create multiple layers of the Transformer block.
    for _ in range(dec_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS,trainable=trainable)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1,trainable=trainable
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS,trainable=trainable)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1,trainable=trainable)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS, trainable=trainable,name="dec_ln_3")(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512, trainable=trainable,name="dec_dense_4")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_4")(x)
    x = layers.Activation('sigmoid')(x)
    
    pre_final = layers.Dense(units=126 * 128 * 2, trainable=trainable,name="dec_dense_5")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_5")(x)
    x = layers.Activation('sigmoid')(x)

    x = layers.Reshape((126, 128, 2))(pre_final)

    x = layers.Conv2D(32, 7, padding='same', trainable=trainable,name="dec_conv_9")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_9")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, 7, padding='same', trainable=trainable,name="dec_conv_10")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_10")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable,name="dec_conv_11")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_11")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x = layers.Conv2D(32, 7, padding='same', trainable=trainable,name="dec_conv_7")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_7")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, 7, padding='same', trainable=trainable,name="dec_conv_8")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_8")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable,name="dec_conv_6")(x)
    x = layers.BatchNormalization(trainable=trainable,name="dec_bn_6")(x)
    decoder_output = layers.Activation('sigmoid')(x)

    return decoder_output
'''


def Encoder(x, feedback_bits, trainable=True):
    B = 4
    with tf.compat.v1.variable_scope('Encoder'):
        x = layers.Conv2D(32, 7, padding='same', trainable=trainable, name="enc_conv_1")(x)
        x = layers.BatchNormalization(trainable=trainable, name="enc_bn_1")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(16, 7, padding='same', trainable=trainable, name="enc_conv_2")(x)
        x = layers.BatchNormalization(trainable=trainable, name="enc_bn_2")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(2, 7, padding='same', trainable=trainable, name="enc_conv_3")(x)
        x = layers.BatchNormalization(trainable=trainable, name="enc_bn_3")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(32, 7, padding='same', trainable=trainable, name="enc_conv_4")(x)
        x = layers.BatchNormalization(trainable=trainable, name="enc_bn_4")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(16, 7, padding='same', trainable=trainable, name="enc_conv_5")(x)
        x = layers.BatchNormalization(trainable=trainable, name="enc_bn_5")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Conv2D(2, 7, padding='same', trainable=trainable, name="enc_conv_6")(x)
        x = layers.BatchNormalization(trainable=trainable, name="enc_bn_6")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits // B), trainable=trainable)(x)
        x = layers.BatchNormalization(trainable=trainable)(x)
        x = layers.Activation('sigmoid')(x)

        encoder_output = QuantizationLayer(B)(x)
    return encoder_output


def Decoder(x, feedback_bits, trainable=True):
    B = 4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.reshape(decoder_input, (-1, int(feedback_bits // B)))
    x = layers.Dense(32256, trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.Activation('sigmoid')(x)
    x = layers.Reshape((126, 128, 2))(x)

    x = layers.Conv2D(32, 7, padding='same', trainable=trainable, name="dec_conv_9")(x)
    x = layers.BatchNormalization(trainable=trainable, name="dec_bn_9")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, 7, padding='same', trainable=trainable, name="dec_conv_10")(x)
    x = layers.BatchNormalization(trainable=trainable, name="dec_bn_10")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable, name="dec_conv_11")(x)
    x = layers.BatchNormalization(trainable=trainable, name="dec_bn_11")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(32, 7, padding='same', trainable=trainable, name="dec_conv_7")(x)
    x = layers.BatchNormalization(trainable=trainable, name="dec_bn_7")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(16, 7, padding='same', trainable=trainable, name="dec_conv_8")(x)
    x = layers.BatchNormalization(trainable=trainable, name="dec_bn_8")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable, name="dec_conv_6")(x)
    x = layers.BatchNormalization(trainable=trainable, name="dec_bn_6")(x)
    decoder_output = layers.Activation('sigmoid')(x)
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
    return {
        "QuantizationLayer":QuantizationLayer,
        "DeuantizationLayer":DeuantizationLayer,
        "Patches":Patches,
        "PatchEncoder":PatchEncoder
    }
