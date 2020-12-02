import matplotlib.pyplot as plt
from gen_data import data_flow
from Network import create_model_2
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import TensorBoard
import keras.backend as K
import keras
import numpy as np
from sklearn.metrics import roc_auc_score,auc,roc_curve
from keras.callbacks import EarlyStopping
data_dir = 'F:/finger/mySOCOFing2'
model_name = 'gender_model.h5'
batch_size = 128
n_classes = 2
img_size = 96
if __name__ == '__main__':
    cmd = input('train or test?\n')
    if cmd == 'train':
        train_sequence, validation_sequence = data_flow(data_dir = data_dir,
                                                        batch_size = batch_size,
                                                        num_classes = n_classes,
                                                        input_size = img_size,
                                                        mode='gender',
                                                        train=True)
        # train_data = train_sequence.__getitem__(300)
        # print(train_data[0][0])
        # print(len(train_sequence))
        #
        # # n = 45997
        # mean_sum = 0
        # std_sum = 0
        # for i in range(len(train_sequence)):
        #     mean_sum += np.array(train_sequence.__getitem__(i)[0][0]).mean()
        #     std_sum += np.array(train_sequence.__getitem__(i)[0][0]).std()
        #
        # for i in range(len(validation_sequence)):
        #     mean_sum += np.array(validation_sequence.__getitem__(i)[0][0]).mean()
        #     std_sum += np.array(validation_sequence.__getitem__(i)[0][0]).std()
        # print(mean_sum / n)
        # print(std_sum/n)

        # sum1 = 0
        # for i in range(len(train_sequence)):
        #
        #     if train_sequence.__getitem__(i)[1][0][0] == 0:
        #        sum1 += 1
        #
        # print(sum1)
        #
        # sum2 = 0
        # for i in range(len(validation_sequence)):
        #     if validation_sequence.__getitem__(i)[1][0][0] == 0:
        #        sum2 += 1
        # print(sum2)
        #
        # sum3 = 0
        # for i in range(len(test_sequence)):
        #     if test_sequence.__getitem__(i)[1][0][0] == 0:
        #         sum3 += 1
        #
        # print(sum3)

        model = create_model_2()
        print(model.summary())


        model.compile(optimizer=Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        tbCallBack = TensorBoard(log_dir="log_file")
        myCallbacks = [
            tbCallBack,
        ]

        history = model.fit_generator(
            generator=train_sequence,
            steps_per_epoch=len(train_sequence),
            epochs=10,
            verbose=1,
            validation_data=validation_sequence,
            validation_steps=len(validation_sequence),
            max_queue_size=10,
            callbacks=myCallbacks,
            shuffle=True
        )
        model.save(model_name)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
    else:
        train_sequence, validation_sequence = data_flow(data_dir=data_dir,
                                                        batch_size=batch_size,
                                                        num_classes=n_classes,
                                                        input_size=img_size,
                                                        mode='gender',
                                                        train=True)
        test_sequence = data_flow(data_dir=data_dir,
                                  batch_size=batch_size,
                                  num_classes=n_classes,
                                  input_size=img_size,
                                  mode='gender',
                                  train=False)
        model = keras.models.load_model(model_name)
        print(model.summary())
        y_true = test_sequence.get_label()
        y_scores = model.predict_generator(test_sequence, verbose=1, steps=len(test_sequence))
        fpr, tpr, thresholds = roc_curve(y_true,y_scores,pos_label=1)
        roc_auc = auc(fpr, tpr)
        print('roc_auc_area:{}'.format(roc_auc))
        ret = model.evaluate_generator(test_sequence, steps=len(test_sequence))
        print('test samples(loss,acc):{}'.format(np.array(ret)))

        plt.figure()
        plt.plot(fpr, tpr,color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC_AUC area for test samples')
        plt.legend(loc="lower right")
        plt.show()



