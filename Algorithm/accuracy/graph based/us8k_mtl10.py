'''
explanation of the name of the file
kd: knowledge distillation
mtl10: multi-task learning, task number is 10

'''


import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from scipy.io import loadmat
import os, shutil
import librosa, random
import pandas as pd
from collections import defaultdict



def feature_extraction():
    '''
    us8k dataset: it takes some time to calculate features,
    so we preprocess it and save time when you have to run many times
    '''

    '''
    official website of us8k: https://urbansounddataset.weebly.com/urbansound8k.html
    we have to first download the us8k dataset
    $ wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
    then extract the compressed files by
    $ tar -xvf UrbanSound8K.tar.gz -C ./

    US8K have varying length, need to pad or truncate them all to a fixed length
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

    conv_1_1_1 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_1_1_1')(pool_0)
    pool_1_1_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1_1')(conv_1_1_1)
    conv_1_1_2 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_1_2')(pool_1_1_1)
    pool_1_1_2 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1_2')(conv_1_1_2)


    conv_1_2_1 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_1_2_1')(pool_0)
    pool_1_2_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_2_1')(conv_1_2_1)
    conv_1_2_2 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_2_2')(pool_1_2_1)
    pool_1_2_2 = MaxPooling2D(pool_size=(2, 2), name='pool_1_2_2')(conv_1_2_2)

    ################  block-2

    flatten_2_1 = Flatten(name='flatten_2_1')(pool_1_1_2)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1 = Dense(480, activation="relu", name='dense_2_1')(drop_2_1)

    flatten_2_2 = Flatten(name='flatten_2_2')(pool_1_1_2)
    drop_2_2 = Dropout(rate=0.5, name='drop_2_2')(flatten_2_2)
    dense_2_2 = Dense(480, activation="relu", name='dense_2_2')(drop_2_2)

    flatten_2_3 = Flatten(name='flatten_2_3')(pool_1_2_2)
    drop_2_3 = Dropout(rate=0.5, name='drop_2_3')(flatten_2_3)
    dense_2_3 = Dense(480, activation="relu", name='dense_2_3')(drop_2_3)

    flatten_2_4 = Flatten(name='flatten_2_4')(pool_1_2_2)
    drop_2_4 = Dropout(rate=0.5, name='drop_2_4')(flatten_2_4)
    dense_2_4 = Dense(480, activation="relu", name='dense_2_4')(drop_2_4)

    flatten_2_5 = Flatten(name='flatten_2_5')(pool_1_2_2)
    drop_2_5 = Dropout(rate=0.5, name='drop_2_5')(flatten_2_5)
    dense_2_5 = Dense(480, activation="relu", name='dense_2_5')(drop_2_5)

    ################ block-3

    dense_3_6 = Dense(2, activation="softmax", name='dense_3_6')(dense_2_1)

    dense_3_8 = Dense(2, activation="softmax", name='dense_3_8')(dense_2_2)

    dense_3_0 = Dense(2, activation="softmax", name='dense_3_0')(dense_2_3)
    dense_3_2 = Dense(2, activation="softmax", name='dense_3_2')(dense_2_3)
    dense_3_3 = Dense(2, activation="softmax", name='dense_3_3')(dense_2_3)
    dense_3_5 = Dense(2, activation="softmax", name='dense_3_5')(dense_2_3)
    dense_3_7 = Dense(2, activation="softmax", name='dense_3_7')(dense_2_3)
    dense_3_9 = Dense(2, activation="softmax", name='dense_3_9')(dense_2_3)

    dense_3_1 = Dense(2, activation="softmax", name='dense_3_1')(dense_2_4)

    dense_3_4 = Dense(2, activation="softmax", name='dense_3_4')(dense_2_5)

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


dataset = 'us8k'
dim_x, dim_y, dim_z = 32, 32, 2 # dimension for corresponding dataset

# please run the following feature_extraction() first if you are running this for the first time
# it pre-computes necessary features that we need to reload repeatedly
# once pre-computed, you can comment this line
# feature_extraction()
# exit()


history_acc1 = []
for test_fold in range(1,11):

    save_path = './checkpoint/{}/test_fold{}/'.format(dataset, test_fold)

    x_train, x_test, y_train, y_test = dataload_dataset()  # for 10-class classification
    trainX, trainY, testX, testY = dataload_mtl() # for binary classification

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
                        # acc1_exist = float(file_or_dir.split('_')[-1])
                        # if acc1_exist < acc1 - 0.01:  # delete folder with lower accuracy
                        shutil.rmtree(save_path + file_or_dir, ignore_errors=True)
            except FileNotFoundError:
                print('\n\nNo previous model exist\n\n')

            history += '{}/{} done: acc1={}\n'.format(i + 1, epochs, acc1)
            print('\n\n', history)
            best_acc1 = acc1
            path = '/'.join(save_path.split('/') + ['acc1_{:.4f}'.format(acc1)])
            model.save(path)
            model.save('./checkpoint/{}/test_fold{}/'.format(dataset, test_fold) + 'best')
