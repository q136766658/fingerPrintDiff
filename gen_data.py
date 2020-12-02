# import necessary libraries
import numpy as np
from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
import os
import cv2
import math
from PIL import Image


class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size, train=False):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path).convert('L')
        img = img.resize((self.img_size[0], self.img_size[1]))
        img = np.array(img)
        img = img.reshape(self.img_size[0], self.img_size[1],1)
        # 数据归一化
        img = np.asarray(img, np.float32) / 255.0
        mean = 0.5697416079349641
        std = 0.3744768065561571
        img[...] -= mean
        img[...] /= std


        return img

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def get_label(self):
        labels = self.x_y[:, 1:]
        labels = np.array(labels).astype(np.float32)
        return labels

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)

# Function to extract labels for both real and altered images
def extract_label(img_path,mode):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')
    # For Real folder
    if(len(etc.split('_')) == 4):
        gender, lr, finger, _ = etc.split('_')
    # For Altered folder
    else:
        gender, lr, finger, _, _ = etc.split('_')

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

    if mode == 'gender':
        return [gender]
    elif mode == 'finger':
        return np.array(finger, dtype=np.uint16)
    elif mode  == 'thumb':
        return [0 if finger == 0 else 1]
    elif mode == 'middle':
        return [0 if finger == 2 else 1]
    else:
        print('The mode must be set gender or finger ')

def smooth_labels(y, smooth_factor=0.1):
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def data_flow(data_dir, batch_size, num_classes, input_size,mode = 'gender',train=True):  # need modify
    img_paths = []
    labels = []
    if train == True:
        paths = [
                os.path.join(data_dir, 'train_Altered-Easy'),
                 os.path.join(data_dir, 'train_Altered-Medium'),
                 os.path.join(data_dir, 'train_Altered-Hard'),
                 os.path.join(data_dir, 'train_Real'),
                os.path.join(data_dir, 'val_Altered-Easy'),
                os.path.join(data_dir, 'val_Altered-Medium'),
                os.path.join(data_dir, 'val_Altered-Hard'),
                os.path.join(data_dir, 'val_Real')
        ]
    else:
        paths = [
            os.path.join(data_dir, 'test_Altered-Easy'),
             os.path.join(data_dir, 'test_Altered-Medium'),
             os.path.join(data_dir, 'test_Altered-Hard'),
            os.path.join(data_dir, 'test_Real')
        ]

    for path in paths:
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                label = extract_label(os.path.join(path, img), mode)
                img_paths.append(img_path)
                labels.append(label)
            except Exception as e:
                pass



    if mode == 'finger':
        labels = np_utils.to_categorical(labels, num_classes)
    # 标签平滑
    # labels = smooth_labels(labels)

    if train == True:
        train_img_paths = img_paths[:41289]
        validation_img_paths = img_paths[41289:]
        train_labels = labels[:41289]
        validation_labels = labels[41289:]

        # train_img_paths, validation_img_paths, train_labels, validation_labels = \
        #     train_test_split(img_paths, labels, test_size=0.1, random_state=30)
        print('total samples: %d, training samples: %d, validation samples: %d' % (
            len(img_paths), len(train_img_paths), len(validation_img_paths)))
        train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size], True)
        validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size], False)
        return train_sequence, validation_sequence
    else:
        test_sequence = BaseSequence(img_paths, labels, batch_size, [input_size, input_size], True)
        print('total test samples: %d' % (
            len(img_paths)))
        return test_sequence
    # # 构造多进程的数据流生成器
    # train_enqueuer = OrderedEnqueuer(train_sequence, use_multiprocessing=True, shuffle=True)
    # validation_enqueuer = OrderedEnqueuer(validation_sequence, use_multiprocessing=True, shuffle=True)
    #
    # # 启动数据生成器
    # n_cpu = multiprocessing.cpu_count()
    # train_enqueuer.start(workers=int(n_cpu * 0.7), max_queue_size=10)
    # validation_enqueuer.start(workers=1, max_queue_size=10)
    # train_data_generator = train_enqueuer.get()
    # validation_data_generator = validation_enqueuer.get()

    # return train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator
