from tkinter import X
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import math

image_size = 128  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 128
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
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

# data_augmentation = keras.Sequential(
#     [
#         layers.Normalization(),
#         # layers.Resizing(image_size, image_size),
#         # layers.RandomFlip("horizontal"),
#         # layers.RandomRotation(factor=0.02),
#         # layers.RandomZoom(
#         #     height_factor=0.2, width_factor=0.2
#         # ),
#     ],
#     name="data_augmentation",
# )
# Compute the mean and the variance of the training data for normalization.
# data_augmentation.layers[0].adapt(x_train)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# CsiNet+(2019)
def Encoder(x, feedback_bits, trainable=True):
    B = 4
    with tf.compat.v1.variable_scope('Encoder'):
        # x = layers.Conv2D(32, 7, padding='same', trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        # x = layers.Conv2D(16, 7, padding='same', trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # y = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        # x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(y)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        # x = layers.Conv2D(32, 7, padding='same', trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        # x = layers.Conv2D(16, 7, padding='same', trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        # x = layers.Add()([x, y])

        # x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x = layers.LeakyReLU(alpha=0.1)(x)
        # # x = layers.Activation('relu')(x)

        # x = layers.Flatten()(x)
        # x = layers.Dense(units=int(feedback_bits // B), trainable=trainable)(x)
        # x = layers.BatchNormalization(trainable=trainable)(x)
        # x = layers.Activation('sigmoid')(x)

        #补0.5至128*128
        x=x-0.5
        x=layers.ZeroPadding2D(padding=(1, 0))(x)
        x=x+0.5
        # Augment data.
        # augmented = data_augmentation(inputs)
        # x = layers.Normalization()(x)
        # Create patches.
        patches = Patches(patch_size)(x)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(128)(features)
        
        encoder_output = logits
    return encoder_output


def Decoder(x,feedback_bits, trainable=True):
    B = 4
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
    y = layers.LeakyReLU(alpha=0.1)(x)
    # x = layers.Activation('relu')(x)

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(y)
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

    x = layers.Add()([x, y])

    x = layers.Conv2D(2, 7, padding='same', trainable=trainable)(x)
    x = layers.BatchNormalization(trainable=trainable)(x)
    x = layers.Activation('sigmoid')(x)
    
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
