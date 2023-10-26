'''
explanation of the name of the file
kd: knowledge distillation
mtl10: multi-task learning, task number is 10
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
from collections import defaultdict
import librosa, random, os, shutil


def feature_extraction():
    '''
    us8k dataset: it takes some time to calculate features,
    so we preprocess it and save time when you have to run many times
    '''

    '''
    official website of us8k: https://urbansounddataset.weebly.com/urbansound8k.html
    we have to first donwload the us8k dataset
    $ wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
    then extract the compressed files by
    $ tar -xvf UrbanSound8K.tar.gz -C ./

    US8K have varying length
    US8K must use 10-fold cross validation for fair comparison
    '''

    # audio files in US8K have varying length, we pad or truncate them all to a fixed length
    pad = lambda a, l: a[0:l] if a.shape[0] > l else np.hstack((a, np.zeros(l - a.shape[0])))

    # parameters
    path = '../dataset/us8k/'
    N_FFT = 1024
    HOP_LENGTH = 1400
    N_MFCC = 32

    df = pd.read_csv(path + 'metadata/UrbanSound8K.csv')
    data = df.to_numpy()  # convert pd dataframe into numpy array

    features = []
    for i in range(data.shape[0]):
        classID = data[i][-2]
        foldID = data[i][-3]  # 1-10
        filename = data[i][0]  # 0-9
        audio, RATE = librosa.load(path + 'audio/fold{}/{}'.format(foldID, filename))

        audio = pad(audio, 88200)  # pad or truncate to a fixed length, here we use 4-second long length
        feature = librosa.feature.mfcc(y=audio, sr=RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
        feature = feature.reshape((N_MFCC, feature.shape[1] // dim_z, dim_z))

        features.append((feature, foldID, classID))

        if i % 200 == 0:
            print('{}/{} processed'.format(i, data.shape[0]))

    PreprocessFeatures = features
    filename = 'PreprocessFeatures.npy'
    np.save(path + filename, PreprocessFeatures)

    print('\nUS8K feature preprocessed and saved!\n\n\n')


def dataload_dataset(chosenType=-1):
    '''
    divide the original dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
                    -1: load the original dataset that has 10-class
    :return: training and testing dataset
    '''

    # load pre-computed features
    path = '../dataset/us8k/'
    filename = 'PreprocessFeatures.npy'
    features = np.load(path + filename, allow_pickle=True)

    x_train, y_train, x_test, y_test = [], [], [], []
    for feature, foldID, classID in features:
        if foldID == test_fold:
            x_test.append(feature)
            y_test.append(classID)
        else:
            x_train.append(feature)
            y_train.append(classID)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    ### make the label list binary with only chosenType being 1 and all other types being 0
    if chosenType != -1:
        y_test = (y_test == chosenType).astype(int)
        y_train = (y_train == chosenType).astype(int)

    return x_train, x_test, y_train, y_test


def get_model():
    '''
    define the multi-task learning model
    :return:
    '''

    ################  block-0

    input_0 = Input(shape=(dim_x, dim_y, dim_z), name='input_0')

    conv_0 = Conv2D(8, kernel_size=(3, 3), activation="relu", name='conv_0')(input_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='pool_0')(conv_0)

    ################  block-1

    conv_1_1 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_1_1')(pool_0)
    pool_1_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1')(conv_1_1)
    conv_1_2 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_2')(pool_1_1)
    pool_1_2 = MaxPooling2D(pool_size=(2, 2), name='pool_1_2')(conv_1_2)

    ################  block-2

    flatten_2_1 = Flatten(name='flatten_2_1')(pool_1_2)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1 = Dense(480, activation="relu", name='dense_2_1')(drop_2_1)

    ################ block-3

    dense_3 = Dense(2, activation="softmax", name='dense_3')(dense_2_1)

    ###############

    model = keras.models.Model(input_0, dense_3)
    return model


def evaluate1(n_class, base_path, x_test, y_test):
    # measure the accuracy of the student model differently
    # we select the class which has the highest probability as the prediction result
    # compare it with the test set
    # if they match, we consider this data point is correctly classified

    pre = []
    for type in range(n_class):
        path = '/'.join(base_path.split('/') + ['type_{}'.format(type)])
        model = tf.keras.models.load_model(path)

        pre.append(model.predict(x_test))

    pre = np.asarray(pre)
    _, n_sample, n_output = pre.shape
    cnt = 0
    for i in range(n_sample):
        cnt += 1 if pre[:, i, 1].argmax() == y_test[i] else 0
    acc = cnt / n_sample

    print(acc)
    return acc


####-----------------------------------------------------------##

dataset = 'us8k'
dim_x, dim_y, dim_z = 32, 32, 2  # input sample dimension for corresponding dataset

# please run the following feature_extraction() first if you are running this for the first time
# it pre-computes necessary features that we need to reload repeatedly
# once pre-computed, you can comment this line
# feature_extraction()
# exit()

history_acc1 = []
for test_fold in range(1, 11):

    save_path = './checkpoint/{}/test_fold{}/'.format(dataset, test_fold)

    for type in range(10):

        model = get_model()
        model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['acc'])

        epochs = 100
        best_acc = 0
        cnt = 0

        path_best = '/'.join(save_path.split('/') + ['type_{}'.format(type)])

        x_train, x_test, y_train, y_test = dataload_dataset(type)
        for epoch in range(epochs):

            print('type={}, epoch={}'.format(type, epoch))
            hist = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
            acc = float('{:.4f}'.format(hist.history['val_acc'][0]))

            if acc > best_acc:  # only save the model if it has better targeted accuracy

                # delete folder with lower accuracy
                try:
                    all = os.listdir(save_path)
                    for file_or_dir in all:
                        if 'type_{}_val_acc_'.format(type) in file_or_dir:
                            # acc1_exist = float(file_or_dir.split('_')[-1])
                            # if acc1_exist < acc1 - 0.01:  # delete folder with lower accuracy
                            shutil.rmtree(save_path + file_or_dir, ignore_errors=True)
                except FileNotFoundError:
                    print('\n\nNo previous model exist\n\n')

                best_acc = acc
                path = '/'.join(save_path.split('/') + ['type_{}_val_acc_{}'.format(type, acc)])
                model.save(path)
                model.save(path_best)
                cnt = 0

                print('\n\n\n', history_acc1, '\n\n\n')

            cnt += 1
            if cnt >= 10 and epoch > 30:  # if no higher acc achieved within 10 consecutive epochs, we exit
                break

    _, x_test, _, y_test = dataload_dataset()
    acc1 = evaluate1(10, save_path, x_test, y_test)
    history_acc1.append((test_fold, '{:.4f}'.format(acc1)))

print('\n\n\n', history_acc1, '\n\n\n')
