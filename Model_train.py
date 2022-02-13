import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 用CPU训练

import numpy as np
import scipy.io as scio
import tensorflow as tf
from tensorflow.keras import optimizers,callbacks,Input,Model
from tensorflow.keras.utils import plot_model
from tensorflow import summary
import tensorflow_addons as tfa
import keras.backend as K
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from Model_define_tf import Encoder, Decoder, NMSE
from datetime import datetime
import shutil
def reset_keras():
    sess = K.get_session()
    K.clear_session()
    sess.close()
    # limit gpu resource allocation
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.visible_device_list = '1'
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     print("Cannot access 'set_memory_growth' - skipping.")
    #     pass 

    # disable arithmetic optimizer
    from tensorflow.core.protobuf import rewriter_config_pb2
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': rewriter_config_pb2.RewriterConfig.OFF
    })

reset_keras()

# parameters
feedback_bits = 512
img_height = 126  # shape=N*126*128*2
img_width = 128
img_channels = 2

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
Encoder_input = Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits, trainable=True)
encoder = Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')
encoder.load_weights('Modelsave/20220214-023008S56.543T0/encoder.h5', by_name=True, skip_mismatch=True)  # 预加载编码器权重
print(encoder.summary())

# decoder model
Decoder_input = Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits, trainable=True)
decoder = Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")
decoder.load_weights('Modelsave/20220214-023008S56.543T0/decoder.h5', by_name=True, skip_mismatch=True)  # 预加载解码器权重
print(decoder.summary())

# autoencoder model
autoencoder_input = Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
# adam_opt = tfa.optimizers.AdamW(learning_rate=0.005,weight_decay = 0.0001)  # 初始学习率为0.001
adam_opt = optimizers.Adam(learning_rate=0.0001)  # 初始学习率为0.001
autoencoder.compile(optimizer=adam_opt, loss='mse', metrics=["acc", score_train])  # 编译模型
print(autoencoder.summary())

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

# 载入训练集
print("loading data set...")
data_load_address = 'train'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']
x_train = x_train.astype('float32')

#训练集分类
def data_Category(x_train,type=0,half_point=50,multi=0.002,dirty=0.0084):
    if type==0: # 不分类
        return x_train
    
    # x_train_abs_l=abs(x_train[:,:half_point,:,0]-0.5+1j*(x_train[:,:half_point,:,1]-0.5))
    x_train_abs_r=abs(x_train[:,half_point:,:,0]-0.5+1j*(x_train[:,half_point:,:,1]-0.5))
    x_train_delay_n=np.mean(x_train_abs_r,axis=2)
    x_train_n=np.mean(x_train_delay_n,axis=1)

    x_train_multi=list()
    x_train_single=list()
    for i,x in enumerate(x_train_n):
        if(x>multi):
            x_train_multi.append(x_train[i])
        else:
            x_train_single.append(x_train[i])
    x_train_multi=np.array(x_train_multi)
    x_train_single=np.array(x_train_single)

    x_train_multi_mean=np.mean(np.mean(abs(x_train_multi[:,:,:,0]-0.5+1j*(x_train_multi[:,:,:,1]-0.5)),axis=1),axis=1)
    x_train_multi_clean=list()
    x_train_multi_dirty=list()
    for i,x in enumerate(x_train_multi_mean):
        if(x<dirty):
            x_train_multi_clean.append(x_train_multi[i])
        else:
            x_train_multi_dirty.append(x_train_multi[i])
    x_train_multi_clean=np.array(x_train_multi_clean)
    x_train_multi_dirty=np.array(x_train_multi_dirty)

    if type==1:
        return x_train_single       # 多径效应不明显的训练集
    elif type==2:
        return x_train_multi_clean  # 多径效应明显清晰的训练集
    elif type==3:
        return x_train_multi_dirty  # 比较模糊的训练集
    else:
        return x_train              # 不分类
data_type = 0
x_train = data_Category(x_train,data_type)

# 数据增强

# x_train_flip=tf.image.flip_up_down(x_train).numpy() #翻转
# x_train=np.concatenate((x_train,x_train_flip))
# x_train=x_train_flip

# half_point=x_train.shape[1]//2 #从中间镜像
# x_train_mir=np.concatenate((x_train[:,:half_point,:,:],x_train_flip[:,half_point:,:,:]), axis=1)
# x_train=np.concatenate((x_train,x_train_mir))
# x_train=x_train_mir

# x_train_noise=gaussian_noise(x_train,0,0.01)#加噪
# x_train=np.concatenate((x_train,x_train_noise))
# x_train=x_train_noise

# 混洗
np.random.shuffle(x_train)

print("x_train",x_train.shape)

# 载入测试集
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']
x_test = x_test.astype('float32')
x_test = data_Category(x_test,data_type)
print("x_test",x_test.shape)

# TensorBoard回调函数
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir_fit = "logs/" + current_time + "/fit"
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir_fit,histogram_freq=1)

# loss停滞时学习率降低回调函数
# lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=20, verbose=1, min_delta=0.0001, min_lr=0.00001)

# 每轮训练完均测试分数并保存最优权重
class bestScoreCallback(callbacks.Callback):
    def __init__(self, x_test, **kwargs):
        self.x_test = x_test
        self.y_test = x_test
        self.best_score = 0
        super(bestScoreCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.y_test = self.model.predict(self.x_test)
        NMSE_test=NMSE(self.x_test, self.y_test)
        self.best_score=Score(NMSE_test)
        print("initial best score:",self.best_score)
        return

    def on_epoch_end(self, epoch, logs=None):
        # self.y_test = self.model.predict(self.x_test)
        # NMSE_test=NMSE(self.x_test, self.y_test)
        # tmp_score = Score(NMSE_test)
        tmp_score = logs['val_score_train']
        if(self.best_score<tmp_score):
            print("update best score from",self.best_score,"to",tmp_score)
            self.best_score=tmp_score
            print("saving Model")
            modelpath = f'./Modelsave/tmp{current_time}T{data_type}/'
            encoder.save(modelpath+"encoder.h5")
            decoder.save(modelpath+"decoder.h5")
            # try:
            #     os.mkdir(modelpath)
            # except:
            #     pass
            # encoder.save_weights(modelpath+"encoder.h5")
            # decoder.save_weights(modelpath+"decoder.h5")
        else:
            print("best score still remain:",self.best_score,",larger than current:",tmp_score)
        return
bsCallback=bestScoreCallback(x_test)

# 早停回调函数
esCBk=EarlyStopping(monitor='val_score_train', patience=100, verbose=1, mode='max', baseline=None, restore_best_weights=False)

my_callbacks = [
    tensorboard_callback,
    bsCallback,
    esCBk,
]

# 训练模型
autoencoder.fit(x=x_train, y=x_train, batch_size=80, epochs=1000, validation_data=(x_test,x_test),callbacks=my_callbacks)

# 评价模型
# y_test = autoencoder.predict(x_test)
# NMSE_test=NMSE(x_test, y_test)
# score_str=str(format(Score(NMSE_test), '.3f'))
# print('The mean NMSE for test set is ' + str(NMSE_test),"score:",score_str)
# y_train = autoencoder.predict(x_train)
# NMSE_train=NMSE(x_train, y_train)
# print('The mean NMSE for train set is ' + str(NMSE_train),"score:",Score(NMSE_train))
score_str=str(format(bsCallback.best_score, '.3f'))

# 保存模型权重、结构图及代码
modelpath = f'./Modelsave/{current_time}S{score_str}T{data_type}/'
try:
    os.rename(f'./Modelsave/tmp{current_time}T{data_type}/', modelpath)
    print("modelpath:",modelpath)
except:
    print("no improvement")

    y_test = autoencoder.predict(x_test)
    NMSE_test=NMSE(x_test, y_test)
    score_str=str(format(Score(NMSE_test), '.3f'))
    print('The mean NMSE for test set is ' + str(NMSE_test),"score:",score_str)

    modelpath = f'./Modelsave/{current_time}S{score_str}T{data_type}/'
    encoder.save(modelpath+"encoder.h5")
    decoder.save(modelpath+"decoder.h5")
    # exit()
# save encoder
# encoder.save(modelpath+"encoder.h5")
try:
    plot_model(encoder,to_file=modelpath+"encoder.png",show_shapes=True)
except:
    plot_model(encoder,to_file=modelpath+"encoder.png",show_shapes=False)
# save decoder
# decoder.save(modelpath+"decoder.h5")
try:
    plot_model(decoder,to_file=modelpath+"decoder.png",show_shapes=True)
except:
    plot_model(decoder,to_file=modelpath+"decoder.png",show_shapes=False)
# save code
shutil.copyfile('./Model_define_tf.py', modelpath+'Model_define_tf.py')

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
    decoded_imgsplo = abs(bsCallback.y_test[i, :, :, 0]-0.5 + 1j*(bsCallback.y_test[i, :, :, 1]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.savefig(modelpath+'csiPlot.png')
# plt.show()