from gen_data import data_flow
from Network import create_model_5
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import TensorBoard
import keras.backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,auc,roc_curve
from keras.callbacks import EarlyStopping
model_name = 'finger_model.h5'
batch_size = 128
n_classes = 5
img_size = 96

if __name__ == '__main__':
    cmd = input('train or test?\n')
    if cmd == 'train':
        train_sequence, validation_sequence = data_flow(data_dir = 'F:/finger/mySOCOFing2',
                                                        batch_size = batch_size,
                                                        num_classes = n_classes,
                                                        input_size = img_size,
                                                        mode='finger',
                                                        train=True)
        test_sequence = data_flow(data_dir = 'F:/finger/mySOCOFing2',
                                  batch_size = batch_size,
                                  num_classes = n_classes,
                                  input_size = img_size,
                                  mode='finger',
                                  train=False)

        model = create_model_5()
        print(model.summary())

        model.compile(optimizer=Adam(1e-3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.metrics_names)

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
        train_sequence, validation_sequence = data_flow(data_dir='F:/finger/mySOCOFing2',
                                                        batch_size=batch_size,
                                                        num_classes=n_classes,
                                                        input_size=img_size,
                                                        mode='finger',
                                                        train=True)
        test_sequence = data_flow(data_dir='F:/finger/mySOCOFing2',
                                  batch_size=batch_size,
                                  num_classes=n_classes,
                                  input_size=img_size,
                                  mode='finger',
                                  train=False)

        model = keras.models.load_model(model_name)
        print(model.summary())
        y_true = test_sequence.get_label()
        y_scores = model.predict_generator(test_sequence, verbose=1, steps=len(test_sequence))

        ret = model.evaluate_generator(test_sequence, steps=len(test_sequence))
        print('test samples(loss,acc):{}'.format(np.array(ret)))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        # micro
        fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_scores.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        finger = ['thumb','index','middle','ring','little']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='{}(area = %0.2f)'.format(finger[i]) % roc_auc[i])

        for i in ['micro','macro']:
            plt.plot(fpr[i], tpr[i], label='{}(area = %0.2f)'.format(i) % roc_auc[i])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC_AUC area for test samples')
        plt.legend(loc="lower right")
        plt.show()
