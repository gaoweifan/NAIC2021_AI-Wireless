import numpy as np
import scipy.io as scio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,callbacks,Input,Model
from tensorflow.keras.utils import plot_model
from tensorflow import summary
import keras.backend as K
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from Model_define_tf import Encoder, Decoder, NMSE
from datetime import datetime
import shutil
# tf.compat.v1.disable_eager_execution()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*9)])
# tf.config.experimental.set_memory_growth(gpus[0],True)
# 固定随机数种子
# import random
# print('GPU', tf.test.is_gpu_available())
# SEED=123456
# import os
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['PYTHONHASHSEED']=str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# parameters
feedback_bits = 512
# img_height = 126  # shape=N*126*128*2
img_height = 126
img_width = 128
img_channels = 2

#添加高斯噪声
def gaussian_noise(img,mean,sigma):
    '''
    此函数将产生高斯噪声加到图片上
    :param img:原图
    :param mean:均值
    :param sigma:标准差
    :return:噪声处理后的图片
    '''

    # img = img/255  #图片灰度标准化

    noise = np.random.normal(mean, sigma, img.shape) #产生高斯噪声
    # print(noise)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    # gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值

#分段线性映射
# left_border=0.49
# right_border=0.51
# mid_k=45
# def y_mid(x):
#     return mid_k*(x-0.5)+0.5

# lr_k=y_mid(left_border)/left_border
# def y_left(x):
#     return lr_k*x

# right_b=y_mid(right_border)-lr_k*right_border
# def y_right(x):
#     return lr_k*x+right_b

# def linearMapping(x):
#     return np.select([x>=right_border, x>=left_border, x<left_border],
#                      [y_right(x),      y_mid(x),       y_left(x)])

# 载入训练集
print("loading data set...")
data_load_address = 'train'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']
x_train = x_train.astype('float32')
# padding_zeros=0.5*np.ones((x_train.shape[0],1,img_width,img_channels),dtype = np.float32)#补0.5至128*128
# x_train = np.concatenate((padding_zeros,x_train,padding_zeros),axis=1)
# x_train_noise=gaussian_noise(x_train,0,0.01)#加噪
# x_train=np.concatenate((x_train,x_train_noise))
# x_train=x_train_noise
# x_train=linearMapping(x_train)#非线性（分段线性）映射
np.random.shuffle(x_train)  # 洗牌
print("x_train",x_train.shape)

# 载入测试集
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']
x_test = x_test.astype('float32')
# padding_zeros=0.5*np.ones((x_test.shape[0],1,img_width,img_channels),dtype = np.float32)#补0.5至128*128
# x_test = np.concatenate((padding_zeros,x_test,padding_zeros),axis=1)
print("x_test",x_test.shape)

# 评价指标
def NMSE_t(x, x_hat):
    x_real = tf.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = tf.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = tf.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = tf.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = tf.complex(x_real - 0.5, x_imag - 0.5)
    x_hat_C = tf.complex(x_hat_real - 0.5, x_hat_imag - 0.5)
    power = tf.reduce_sum(tf.abs(x_C) ** 2, axis=1)
    mse = tf.reduce_sum(tf.abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = tf.reduce_mean(mse / power)
    return nmse

def Score(NMSE):
    score = (1 - NMSE) * 100
    return score

def score_train(y_true, y_pred):
    return Score(NMSE_t(y_true, y_pred))

# 建立模型
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# encoder model
latent_dim = 6

encoder_inputs = Input(shape=(img_height, img_width, img_channels))
# x = layers.Conv2D(32, 7, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv2D(64, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(128, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(64, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(512, activation="relu")(x)
x = layers.Conv2D(6, 3, padding='same', activation='relu')(encoder_inputs)
x = layers.Conv2D(6, 3, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(units=int(512 // 4), activation='sigmoid')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.load_weights("Modelsave/20220121-173329S52.835/encoder.h5",by_name=True, skip_mismatch=True)
encoder.summary()

# decoder model
latent_inputs = Input(shape=(latent_dim,))
# x = tf.reshape(latent_inputs, (-1, int(512//4)))
x = layers.Dense(32256, activation='sigmoid')(latent_inputs)
x_ini = layers.Reshape((126, 128, 2))(x)
for i in range(3):
    x = layers.Conv2D(8, 3, padding='SAME', activation='relu')(x_ini)
    x = layers.Conv2D(16, 3, padding='SAME', activation='relu')(x)
    x = layers.Conv2D(2, 3, padding='SAME', activation='relu')(x)
    x_ini = layers.Add()([x_ini, x])
decoder_outputs = layers.Conv2D(2, 3, padding='SAME',activation="sigmoid")(x_ini)
# x = layers.Dense(int((img_height/16) * (img_width/16) * 64), activation="relu")(latent_inputs)
# x = layers.Reshape((int(img_height/16), int(img_width/16), 64))(x)
# x = layers.Conv2DTranspose(128, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(64, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 7, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(16, 7, activation="relu", strides=2, padding="same")(x)
# decoder_outputs = layers.Conv2DTranspose(img_channels, 3, activation="sigmoid", padding="same")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.load_weights("Modelsave/20220121-173329S52.835/decoder.h5",by_name=True, skip_mismatch=True)
decoder.summary()

# autoencoder model
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )-11000
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = 100000*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# 训练模型
vae.fit(x_train, epochs=200, batch_size=128)

# 评价模型
_,_,y_test_e = encoder.predict(x_test)
y_test = decoder.predict(y_test_e)
# NMSE_test=NMSE(x_test[:,1:127,:,:], y_test[:,1:127,:,:])
NMSE_test=NMSE(x_test, y_test)
score_str=str(format(Score(NMSE_test), '.3f'))
print('The mean NMSE for test set is ' + str(NMSE_test),"score:",score_str)
encoder.save(f"vae_encoder_{score_str}.h5")
decoder.save(f"vae_decoder_{score_str}.h5")
# y_train = autoencoder.predict(x_train)
# NMSE_train=NMSE(x_train, y_train)
# print('The mean NMSE for train set is ' + str(NMSE_train),"score:",Score(NMSE_train))

# # 保存模型权重、结构图及代码
# # save encoder
# modelpath = f'./Modelsave/{current_time}S{score_str}/'
# encoder.save(modelpath+"encoder.h5")
# try:
#     plot_model(encoder,to_file=modelpath+"encoder.png",show_shapes=True)
# except:
#     plot_model(encoder,to_file=modelpath+"encoder.png",show_shapes=False)
# # save decoder
# decoder.save(modelpath+"decoder.h5")
# try:
#     plot_model(decoder,to_file=modelpath+"decoder.png",show_shapes=True)
# except:
#     plot_model(decoder,to_file=modelpath+"decoder.png",show_shapes=False)
# # save code
# shutil.copyfile('./Model_define_tf.py', modelpath+'Model_define_tf.py')

# 以下是可视化作图部分
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1)
    # x_testplo = abs(x_test[i, 1:127, :, 0]-0.5 + 1j*(x_test[i, 1:127, :, 1]-0.5))
    x_testplo = abs(x_test-0.5 + 1j*(x_test-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    # decoded_imgsplo = abs(y_test[i, 1:127, :, 0]-0.5 + 1j*(y_test[i, 1:127, :, 1]-0.5))
    decoded_imgsplo = abs(y_test-0.5 + 1j*(y_test-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.savefig('vae_csiPlot.png')
# plt.show()