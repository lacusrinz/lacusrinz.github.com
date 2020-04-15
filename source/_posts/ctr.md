---
title: 使用deepCTR库实践
date: 2020-02-09 20:53:58
tags: [CTR, tensorflow, deepFM]
categories: 机器学习应用
---

## 目标
使用deepCTR库快速完成一个deepFM模型训练
<!--more-->

[DeepCTR](https://github.com/shenweichen/DeepCTR)

## 开发环境
Google Driver + Google Colaboratory
在Driver中创建`ctr.ipynb`用Colaboratory打开

## 数据集
数据集我们使用Kaggle上比赛 [Criteo Display Advertising Challenge Predict click-through rates on display ads](https://www.kaggle.com/c/criteo-display-ad-challenge) 的数据集
Kaggle网站上的数据下载地址已失效，下载地址[点此](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz)

### 准备原始数据
连入Google Driver
```python
from google.colab import drive

drive.mount("./gdrive", force_remount=True)

%cd "./gdrive/My Drive/deepCTR/"
```

下载criteo数据集
```python
%cd raw
!wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
```
在Google Driver中解压
train.txt 11G
test.txt 1.4G
源文件太大，我们取前100w行做训练
```python
!head -n 1000000 train.txt > train_sub100w.txt
```

## 配置环境
因为Colaboratory环境有有GPU，所有我们安装DeepCTR的GPU版本
```python
pip install deepctr[gpu]
```

```python
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

%tensorflow_version 2.x
```

### 数据预处理
数据集本身没有column名，手动加上，
label是结果
I1-I13是连续型的类型
C1-C26是离散型的数据
所有数据都hash脱敏

**导入csv并补0**
```python
columns = ['label','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']
data = pd.read_csv('./raw/train_sub100w.txt', names=columns, sep='\t')
print(data)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']
print(data)
```
输出
![1-1](1-1.png)


**数据编码和归一化**
LabelEncoder可以将标签分配一个 0—n classes之间的编码 fit_transform(self, y) Fit label encoder and return encoded labels

MinMaxScaler将属性缩放到一个指定的最大和最小值（通常是1-0）之间
```python
for feat in sparse_features:
  lbe = LabelEncoder()
  data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
print(data)
```
输出
![1-2](1-2.png)
可以看到连续型的I字段都已经呗归一化到0-1之间
离散型的C字段每个hash值都被分配了一个编号，例如C25的第一行第二行都是e8b83407，编码后都是58

**获取feature name**
```python
from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

# count #unique features for each sparse field,and record dense feature field name

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) 
              for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,) 
              for feat in dense_features
              ]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

print(feature_names)
```
输出
['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']

**构筑input data**
```python
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}
for name in feature_names:
  print(name)
  s = []
  for i in range(1,10):
    s.append(train_model_input[name].tolist()[i])
  print(s)
```
输出每个column包含的前9个值（节选）
![1-3](1-3.png)

## 训练
设置输出地址
```python
output_path = "./output"
target_path = "./output/checkpoint_weights.hdf5"
```
**Keras fit方法的定义**
>fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

>x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array。如果模型的每个输入都有名字，则可以传入一个字典，将输入名与其输入数据对应起来。

>y：标签，numpy array。如果模型有多个输出，可以传入一个numpy array的list。如果模型的输出拥有名字，则可以传入一个字典，将输出名与其标签对应起来。

>batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

>epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch

>verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

>callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数

>validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之后，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。

>validation_data：形式为（X，y）或（X，y，sample_weights）的tuple，是指定的验证集。此参数将覆盖validation_spilt。

>shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。

>class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）。该参数在处理非平衡的训练数据（某些类的训练样本数很少）时，可以使得损失函数对样本数不足的数据更加关注。

>sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。

>initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。

>steps_per_epoch: 一个epoch包含的步数（每一步是一个batch的数据送入），当使用如TensorFlow数据Tensor之类的输入张量进行训练时，默认的None代表自动分割，即数据集样本数/batch样本数。

>validation_steps: 仅当steps_per_epoch被指定时有用，在验证集上的step总数。

```python
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 4.Define Model,train,predict and evaluate

model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(opt, "binary_crossentropy", metrics=['binary_crossentropy', 'accuracy'], )

tbCallBack = TensorBoard(
              log_dir=output_path, 
              histogram_freq=10, 
              write_graph=True, 
              write_images=False, 
              update_freq="epoch")
ckCallBack = ModelCheckpoint(
                filepath=target_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1)
esCallBack = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-8,
                patience=5,
                restore_best_weights=True,
                verbose=1)
rrCallBack = ReduceLROnPlateau(
                monitor="val_loss",
                min_delta=1e-8,
                factor=0.2,
                patience=3,
                verbose=1)

history = model.fit(train_model_input, train[target].values,
                        batch_size=512, epochs=50, verbose=1, validation_split=0.2, callbacks=[tbCallBack, ckCallBack, esCallBack, rrCallBack], )
pred_ans = model.predict(test_model_input, batch_size=512)

print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
```

输出
![1-4](1-4.png)

第一个epoch即可达到最佳效果，继续训练出现过拟合，**正常现象**



