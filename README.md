# NAIC2021

### 介绍
NAIC2021比赛，AI+无线通信赛道

### 软件架构
#### Model_define_tf.py
为定义模型的代码，如encoder、decoder、量化层等，提交时就提交这个和训练好的模型权重
#### Model_train.py
为训练模型的代码
#### Modelsave
文件夹内为历次训练的模型权重、自动生成的模型结构图、CSI图像输入输出的效果图。
按照时间及得分划分子文件夹，格式为modelpath = f'./Modelsave/{current_time}S{score_str}/'
TODO:也可以将本次训练使用的Model_define_tf.py一并放入（通过代码自动完成）
#### logs
文件夹内为tensorboard保存的记录文件，可以通过tensorboard --logdir logs启用可视化网页查看
#### train
文件夹内为数据集，8000个训练集样本和2000个测试集样本，样本形状为(None,126,128,2)
### 环境要求
1.  python==3.9.7
2.  tensorflow-gpu==2.6.0
3.  keras==2.6.0
4.  numpy==1.21.2
5.  cuda>=11.3.1
6.  cudnn>=8.2.1

### 使用说明
1.  先按照要求搭建运行环境，未提及的模块在运行时若报错请自己查询添加
2.  在Model_define_tf.py内部定义模型结构
3.  运行Model_train.py开始训练，训练完后会自动保存训练权重、模型结构图、CSI图像效果图到相应文件夹内
4.  提交时请提交Model_define_tf.py及两个训练好的encoder.h5、decoder.h5权重