import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

# 从图像标签中提取性别
def extract_label(img_path, train=True):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')
    # For Altered folder
    if train:
        gender, lr, finger, _, _ = etc.split('_')
    # For Real folder
    else:
        gender, lr, finger, _ = etc.split('_')

    gender = 0 if gender == 'M' else 1
    lr = 0 if lr == 'Left' else 1

    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
    return np.array([gender], dtype=np.uint16)


# 加载数据
img_size = 96
# Function to iterate through all the images
def loading_data(path, train):
    print("loading data from: ", path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img), train)
            data.append([label[0], img_resize])
        except Exception as e:
            pass
    data
    return data

# 分配各种目录，并在每个目录上使用loading_data 函数
Real_path = "F:/finger/SOCOFing/Real"
Easy_path = "F:/finger/SOCOFing/Altered/Altered-Easy"
Medium_path = "F:/finger/SOCOFing/Altered/Altered-Medium"
Hard_path = "F:/finger/SOCOFing/Altered/Altered-Hard"

Easy_data = loading_data(Easy_path, train=True)
Medium_data = loading_data(Medium_path, train=True)
Hard_data = loading_data(Hard_path, train=True)
test = loading_data(Real_path, train=False)

data = np.concatenate([Easy_data, Medium_data, Hard_data], axis=0)

del Easy_data, Medium_data, Hard_data

# 随机化训练数据data和测试数据test数组
import random
random.shuffle(data)
random.shuffle(test)

# 分离图像数组和标签
img, labels = [], []
for label, feature in data:
    labels.append(label)
    img.append(feature)
train_data = np.array(img).reshape(-1, img_size, img_size, 1)
train_data = train_data / 255.0
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(labels, num_classes=2)
del data


# 构建模型网络结构
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from tensorflow.keras import layers
# from tensorflow.keras import optimizers
#
# model = Sequential([
#     Conv2D(
#         32,
#         3,
#         padding='same',
#         activation='relu',
#         kernel_initializer='he_uniform',
#         input_shape=[
#             96,
#             96,
#             1]),
#     MaxPooling2D(2),
#     Conv2D(
#         32,
#         3,
#         padding='same',
#         kernel_initializer='he_uniform',
#         activation='relu'),
#     MaxPooling2D(2),
#     Flatten(),
#     Dense(128, kernel_initializer='he_uniform', activation='relu'),
#     Dense(2, activation='softmax'),
# ])
#
#
# # 编译模型
# model.compile(
#     optimizer=optimizers.Adam(1e-3),
#     loss='categorical_crossentropy',
#     metrics=['accuracy'])

#Import necessary libraries
from keras import layers
from keras import models
from keras.models import Model
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import TensorBoard, Callback
import tensorflow as tf

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu',input_shape=(96, 96, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

print(model.summary())

model.compile(optimizer = Adam(1e-1), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # 创建一个保存模型权重的回调
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                    save_weights_only=True,
#                                                    verbose=1)
# # 防止过拟合
# early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=10)

history = model.fit(train_data, train_labels, batch_size=128, epochs=15,
                    validation_split=0.2, verbose=1)


