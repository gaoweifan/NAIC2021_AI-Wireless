import numpy as np
import scipy.io as scio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow import summary
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from Model_define_tf import Encoder, Decoder, NMSE
from datetime import datetime
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
img_height = 126  # shape=N*126*128*2
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
# encoder model
Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits)
encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')
# encoder.load_weights('Modelsave/20220121-173329S52.835/encoder.h5')  # 预加载编码器权重
print(encoder.summary())

# decoder model
Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits)
decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")
# decoder.load_weights('Modelsave/20220121-173329S52.835/decoder.h5')  # 预加载解码器权重
print(decoder.summary())

# autoencoder model
autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse', metrics=["acc", score_train])  # 编译模型
print(autoencoder.summary())

# TensorBoard回调函数
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir_fit = "logs/" + current_time + "/fit"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir_fit,histogram_freq=1)

# 学习率记录回调函数
# logdir_lr = "logs/" + current_time + "/lr"
# summary_writer = summary.create_file_writer(logdir_lr)
# def lr_sche(epoch):
#     # 每隔100个epoch，学习率减小为原来的1/10
#     # if epoch % 100 == 0 and epoch != 0:
#     #     lr = K.get_value(model.optimizer.lr)
#     #     K.set_value(model.optimizer.lr, lr * 0.1)
#     #     print("lr changed to {}".format(lr * 0.1))
#     lr = K.get_value(autoencoder.optimizer.lr)
#     print("learning rate:",lr)
#     with summary_writer.as_default():
#         summary.scalar('leaning_rate', lr, step=epoch)
#     return lr
# lr_callback = LearningRateScheduler(lr_sche)

# 训练模型
autoencoder.fit(x=x_train, y=x_train, batch_size=16, epochs=200, validation_split=0.2,callbacks=[tensorboard_callback])

# 评价模型
y_test = autoencoder.predict(x_test)
NMSE_test=NMSE(x_test, y_test)
score_str=str(format(Score(NMSE_test), '.3f'))
print('The mean NMSE for test set is ' + str(NMSE_test),"score:",score_str)
# y_train = autoencoder.predict(x_train)
# NMSE_train=NMSE(x_train, y_train)
# print('The mean NMSE for train set is ' + str(NMSE_train),"score:",Score(NMSE_train))

# 保存模型
# save encoder
modelpath = f'./Modelsave/{current_time}S{score_str}/'
encoder.save(modelpath+"encoder.h5")
plot_model(encoder,to_file=modelpath+"encoder.png",show_shapes=True)
# save decoder
decoder.save(modelpath+"decoder.h5")
plot_model(decoder,to_file=modelpath+"decoder.png",show_shapes=True)


# 以下是可视化作图部分
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1)
    x_testplo = abs(x_test[i, :, :, 0]-0.5 + 1j*(x_test[i, :, :, 1]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(y_test[i, :, :, 0]-0.5 + 1j*(y_test[i, :, :, 1]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.savefig(f'./Modelsave/{current_time}S{score_str}/plot.png')
# plt.show()