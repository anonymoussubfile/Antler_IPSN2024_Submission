'''
explanation of the name of the file
kd: knowledge distillation
mtl10: multi-task learning, task number is 10

'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import os, shutil, librosa, random
import pandas as pd
from collections import defaultdict


def dataload_mtl():
    '''
    :return: data for training MTL models
    '''
    trainX = dataload_dataset(0)[0]
    testX = dataload_dataset(0)[1]

    trainY = []
    testY = []
    for i in range(10):
        trainY.append(dataload_dataset(i)[2])
        testY.append(dataload_dataset(i)[3])

    return trainX, trainY, testX, testY


def dataload_dataset(chosenType=-1):
    '''
    divide the original dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
                    -1: load the original dataset that has 10-class
    :return: training and testing dataset
    '''

    '''
    for esc you have to first download the original esc dataset
    there used to be esc10 and esc50 datasets and they were separate
    now esc10 has been incorporated into esc50, we have to download esc50 first
    and then extract esc10 from the downloaded esc50

    to download esc50 dataset, use the following command
    $ wget https://github.com/karoldvl/ESC-50/archive/master.zip
    $ unzip master.zip -d ./
    then we rename folder name from ESC-50-master to esc

    reference:
    https://github.com/karolpiczak/paper-2015-esc-dataset/blob/master/Notebook/ESC-Dataset-for-Environmental-Sound-Classification.ipynb
    '''

    # parameters
    RATE = 44100
    N_FFT = 1024
    HOP_LENGTH = 1024
    N_MFCC = dim_x

    df = pd.read_csv('../dataset/esc/meta/esc50.csv')
    df = df.loc[df['esc10'] == True]  # we only keep esc10
    df = df[['filename', 'target']]  # only keep two columns

    files = df.to_numpy()  # convert pd dataframe into numpy array
    dataset = defaultdict(list)  # save files into a dict
    for filename, target in files:
        dataset[target].append(filename)

    # ESC-10: every class has exactly 40 samples
    # each sample is exactly 5-second long
    # we use 80% for training, 32 samples from each class
    # 20% for testing

    x_train, y_train, x_test, y_test = [], [], [], []
    label = 0  # relabel classes to 1-9
    for key in dataset.keys():
        random.shuffle(dataset[key])  # shuffle the filename list
        for cnt, file in enumerate(dataset[key]):
            audio, _ = librosa.load('../dataset/esc/audio/' + file, sr=RATE)
            feature = librosa.feature.mfcc(y=audio, sr=RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
            feature = feature.reshape((N_MFCC, feature.shape[1] // dim_z, dim_z))

            if cnt < 32:
                x_train.append(feature)
                y_train.append(label)
            else:
                x_test.append(feature)
                y_test.append(label)

            cnt += 1

        label += 1

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    ### make the label list binary with only chosenType being 1 and all other types being 0
    if chosenType != -1:
        y_test = (y_test == chosenType).astype(int)
        y_train = (y_train == chosenType).astype(int)

    return x_train, x_test, y_train, y_test
    # return x_train[:1000], x_test[:1000], y_train[:1000], y_test[:1000]  # for debug


def get_model():
    '''
    define the multi-task learning model
    :return:
    '''

    ################  block-0

    input_0 = Input(shape=(dim_x, dim_y, dim_z), name='input_0')

    conv_0 = Conv2D(8, kernel_size=(12, 6), activation="relu", name='conv_0')(input_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='pool_0')(conv_0)

    ################  block-1

    conv_1_1 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_1_1')(pool_0)
    pool_1_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1')(conv_1_1)

    conv_1_2 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_1_2')(pool_0)
    pool_1_2 = MaxPooling2D(pool_size=(2, 2), name='pool_1_2')(conv_1_2)

    conv_1_3 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_1_3')(pool_0)
    pool_1_3 = MaxPooling2D(pool_size=(2, 2), name='pool_1_3')(conv_1_3)

    ################  block-2

    conv_2_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_2_1')(pool_1_1)
    pool_2_1 = MaxPooling2D(pool_size=(2, 2), name='pool_2_1')(conv_2_1)
    flatten_2_1 = Flatten(name='flatten_2_1')(pool_2_1)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1_1 = Dense(128, activation="relu", name='dense_2_1_1')(drop_2_1)
    dense_2_1_2 = Dense(64, activation="relu", name='dense_2_1_2')(dense_2_1_1)

    conv_2_2 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_2_2')(pool_1_2)
    pool_2_2 = MaxPooling2D(pool_size=(2, 2), name='pool_2_2')(conv_2_2)
    flatten_2_2 = Flatten(name='flatten_2_2')(pool_2_2)
    drop_2_2 = Dropout(rate=0.5, name='drop_2_2')(flatten_2_2)
    dense_2_2_1 = Dense(128, activation="relu", name='dense_2_2_1')(drop_2_2)
    dense_2_2_2 = Dense(64, activation="relu", name='dense_2_2_2')(dense_2_2_1)

    conv_2_3 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_2_3')(pool_1_2)
    pool_2_3 = MaxPooling2D(pool_size=(2, 2), name='pool_2_3')(conv_2_3)
    flatten_2_3 = Flatten(name='flatten_2_3')(pool_2_3)
    drop_2_3 = Dropout(rate=0.5, name='drop_2_3')(flatten_2_3)
    dense_2_3_1 = Dense(128, activation="relu", name='dense_2_3_1')(drop_2_3)
    dense_2_3_2 = Dense(64, activation="relu", name='dense_2_3_2')(dense_2_3_1)

    conv_2_4 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_2_4')(pool_1_3)
    pool_2_4 = MaxPooling2D(pool_size=(2, 2), name='pool_2_4')(conv_2_4)
    flatten_2_4 = Flatten(name='flatten_2_4')(pool_2_4)
    drop_2_4 = Dropout(rate=0.5, name='drop_2_4')(flatten_2_4)
    dense_2_4_1 = Dense(128, activation="relu", name='dense_2_4_1')(drop_2_4)
    dense_2_4_2 = Dense(64, activation="relu", name='dense_2_4_2')(dense_2_4_1)

    ################ block-3

    dense_3_1 = Dense(2, activation="softmax", name='dense_3_1')(dense_2_1_2)

    dense_3_0 = Dense(2, activation="softmax", name='dense_3_0')(dense_2_2_2)
    dense_3_3 = Dense(2, activation="softmax", name='dense_3_3')(dense_2_2_2)
    dense_3_4 = Dense(2, activation="softmax", name='dense_3_4')(dense_2_2_2)
    dense_3_6 = Dense(2, activation="softmax", name='dense_3_6')(dense_2_2_2)
    dense_3_7 = Dense(2, activation="softmax", name='dense_3_7')(dense_2_2_2)
    dense_3_8 = Dense(2, activation="softmax", name='dense_3_8')(dense_2_2_2)
    dense_3_9 = Dense(2, activation="softmax", name='dense_3_9')(dense_2_2_2)

    dense_3_2 = Dense(2, activation="softmax", name='dense_3_2')(dense_2_3_2)

    dense_3_5 = Dense(2, activation="softmax", name='dense_3_5')(dense_2_4_2)

    ###############

    model = keras.models.Model(input_0, [dense_3_0,
                                         dense_3_1,
                                         dense_3_2,
                                         dense_3_3,
                                         dense_3_4,
                                         dense_3_5,
                                         dense_3_6,
                                         dense_3_7,
                                         dense_3_8,
                                         dense_3_9])
    return model


def evaluate1(model, x_test, y_test):
    # measure the accuracy of the student model differently
    # we select the class which has the highest probability as the prediction result
    # compare it with the test set
    # if they match, we consider this data point is correctly classified

    pre = model.predict(x_test)
    pre = np.asarray(pre)

    n_class, n_sample, n_output = pre.shape
    cnt = 0
    for i in range(n_sample):
        cnt += 1 if pre[:, i, 1].argmax() == y_test[i] else 0
    acc = cnt / n_sample

    # print(acc)
    return acc


def evaluate2(model, x_test, y_test):
    # measure the accuracy of the student model differently
    # we check the output of all tasks individually
    # only when there is only one class saying Yes and all others saying No
    # plus, the class saying Y must match with the true class
    # then, we consider this data point is correctly classified

    pre = model.predict(x_test)
    pre = np.asarray(pre)

    n_class, n_sample, n_output = pre.shape
    cnt = 0
    for i in range(n_sample):
        yes = pre[:, i, 1] > pre[:, i, 0]
        cnt += 1 if np.sum(yes) == 1 and yes.argmax() == y_test[i] else 0
    acc = cnt / n_sample

    # print(acc)
    return acc


####-----------------------------------------------------------##

dataset = 'esc10'
save_path = './checkpoint/{}/'.format(dataset)
dim_x, dim_y, dim_z = 32, 108, 2  # dimension for corresponding dataset

x_train, x_test, y_train, y_test = dataload_dataset()  # for 10-class classification
trainX, trainY, testX, testY = dataload_mtl()  # for binary classification

# model = tf.keras.models.load_model('./checkpoint/')
model = get_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['acc'])

epochs = 100
best_acc1 = 0
history = ''
for i in range(epochs):
    model.fit(trainX, trainY, epochs=1)
    acc1 = evaluate1(model, x_test, y_test)
    print('\n{}/{} is done, acc1 = {:.4f}\n'.format(i, epochs, acc1))
    if acc1 > best_acc1:  # only save the model if it has better targeted accuracy

        # delete folder with lower accuracy
        try:
            all = os.listdir(save_path)
            for file_or_dir in all:
                if 'acc' in file_or_dir:
                    acc1_exist = float(file_or_dir.split('_')[-1])
                    if acc1_exist < acc1 - 0.01:  # delete folder with lower accuracy
                        shutil.rmtree(save_path + file_or_dir, ignore_errors=True)
        except FileNotFoundError:
            print('\n\nNo previous model exist\n\n')

        history += '{}/{} done: acc1={}\n'.format(i + 1, epochs, acc1)
        print('\n\n', history)
        best_acc1 = acc1
        path = '/'.join(save_path.split('/') + ['acc1_{:.4f}'.format(acc1)])
        model.save(path)
        model.save('./checkpoint/{}/'.format(dataset) + 'best')
