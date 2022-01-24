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


NUM_CLASSES = 1000


def swish(x):
    return x * tf.nn.sigmoid(x)


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = layers.GlobalAveragePooling2D()
        self.reduce_conv = layers.Conv2D(filters=self.num_reduced_filters,
                                         kernel_size=(1, 1),
                                         strides=1,
                                         padding="same")
        self.expand_conv = layers.Conv2D(filters=input_channels,
                                         kernel_size=(1, 1),
                                         strides=1,
                                         padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


class MBConv(layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = layers.Conv2D(filters=in_channels * expansion_factor,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.bn1 = layers.BatchNormalization()
        self.dwconv = layers.DepthwiseConv2D(kernel_size=(k, k),
                                             strides=stride,
                                             padding="same")
        self.bn2 = layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = layers.Conv2D(filters=out_channels,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.bn3 = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = layers.add([x, inputs])
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'stride': self.stride,
            'drop_connect_rate': self.drop_connect_rate
        })
        return config


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block


# More details about the neural networks can be found in [1].
# [1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback,"
# in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
def Encoder(x, feedback_bits, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, drop_connect_rate=0.2):
    B = 4
    # conv1 = layers.Conv2D(filters=round_filters(32, width_coefficient),
    #                       kernel_size=(3, 3),
    #                       strides=2,
    #                       padding="same")
    # bn1 = layers.BatchNormalization()
    # block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
    #                             out_channels=round_filters(16, width_coefficient),
    #                             layers=round_repeats(1, depth_coefficient),
    #                             stride=1,
    #                             expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
    # block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
    #                             out_channels=round_filters(24, width_coefficient),
    #                             layers=round_repeats(2, depth_coefficient),
    #                             stride=2,
    #                             expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
    # block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
    #                             out_channels=round_filters(40, width_coefficient),
    #                             layers=round_repeats(2, depth_coefficient),
    #                             stride=2,
    #                             expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
    # block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
    #                             out_channels=round_filters(80, width_coefficient),
    #                             layers=round_repeats(3, depth_coefficient),
    #                             stride=2,
    #                             expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
    # block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
    #                             out_channels=round_filters(112, width_coefficient),
    #                             layers=round_repeats(3, depth_coefficient),
    #                             stride=1,
    #                             expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
    # block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
    #                             out_channels=round_filters(192, width_coefficient),
    #                             layers=round_repeats(4, depth_coefficient),
    #                             stride=2,
    #                             expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
    # block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
    #                             out_channels=round_filters(320, width_coefficient),
    #                             layers=round_repeats(1, depth_coefficient),
    #                             stride=1,
    #                             expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
    #
    # conv2 = layers.Conv2D(filters=round_filters(1280, width_coefficient),
    #                       kernel_size=(1, 1),
    #                       strides=1,
    #                       padding="same")
    # bn2 = layers.BatchNormalization()
    # pool = layers.GlobalAveragePooling2D()
    # dropout = layers.Dropout(rate=dropout_rate)
    # fc = layers.Dense(units=NUM_CLASSES,
    #                   activation=tf.keras.activations.softmax)
    with tf.compat.v1.variable_scope('Encoder'):
        x = layers.Conv2D(2, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(2, 3, padding='same', activation='relu')(x)

        # x = layers.Conv2D(2, 4, padding='valid', activation='relu')(x)
        # x = layers.Conv2D(2, 4, padding='valid', activation='relu')(x)
        # x = layers.Conv2D(2, 5, padding='valid', activation='relu')(x)
        # x = layers.Conv2D(2, 3, padding='valid', activation='relu')(x)

        # x = conv1(x)
        # x = bn1(x)
        # x = swish(x)
        #
        # x = block1(x)
        # x = block2(x)
        # x = block3(x)
        # x = block4(x)
        # x = block5(x)
        # x = block6(x)
        # x = block7(x)
        #
        # x = conv2(x)
        # x = bn2(x)
        # x = swish(x)
        # x = pool(x)
        # x = dropout(x)
        # x = fc(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits // B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output


def Decoder(x,feedback_bits):
    B = 4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.reshape(decoder_input, (-1, int(feedback_bits//B)))
    x = layers.Dense(32256, activation='sigmoid')(x)
    x_ini = layers.Reshape((126, 128, 2))(x)
    for i in range(3):
        x = layers.Conv2D(8, 3, padding='SAME', activation='relu')(x_ini)
        x = layers.Conv2D(16, 3, padding='SAME', activation='relu')(x)
        x = layers.Conv2D(2, 3, padding='SAME', activation='relu')(x)
        x_ini = keras.layers.Add()([x_ini, x])
    decoder_output = layers.Conv2D(2, 3, padding='SAME',activation="sigmoid")(x_ini)
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
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer,"MBConv":MBConv,"SEBlock":SEBlock}
