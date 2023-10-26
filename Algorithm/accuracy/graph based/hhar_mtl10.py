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
import os, shutil, csv
import librosa, random
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler



def dataload_mtl():
    '''
    :return: data for training MTL models
    '''
    trainX = dataload_dataset(0)[0]
    testX = dataload_dataset(0)[1]

    trainY = []
    testY = []
    for i in range(n_task):
        trainY.append(dataload_dataset(i)[2])
        testY.append(dataload_dataset(i)[3])

    return trainX, trainY, testX, testY


def create_segments_and_labels(data, window, step):

    # x, y, z acceleration as features
    N_FEATURES = 3

    segments = []
    labels = []
    for i in range(0, len(data) - window, step):
        xs = data[:,2][i: i + window]
        ys = data[:,3][i: i + window]
        zs = data[:,4][i: i + window]

        # Retrieve the most often used label in this segment
        bins = np.asarray((data[:,1][i: i + window]), dtype = int)
        counts = np.bincount(bins)
        label = np.argmax(counts)

        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments).reshape(-1, 10, 10, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def dataload_dataset(chosenType=-1):
    '''
    divide the original dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
                    -1: load the original dataset that has 10-class
    :return: training and testing dataset
    '''

    path = '../dataset/hhar/'
    filename = 'WISDM_ar_v1.1_raw.txt'

    gtFile = open(path + filename)
    gtReader = csv.reader(gtFile, delimiter=',')
    labels = {'Downstairs': 0, 'Jogging': 1, 'Sitting': 2, 'Standing': 3, 'Upstairs': 4, 'Walking': 5}

    data = []
    for row in gtReader:
        '''
        column_names = ['user-id',
                      'activity',
                      'timestamp',
                      'x-axis',
                      'y-axis',
                      'z-axis']
        example: ['33', 'Jogging', '49106062271000', '5.012288', '11.264028', '0.95342433;']
        '''
        try:
            user = int(row[0])
            label = labels[row[1]]
            x = float(row[3])
            y = float(row[4])
            z = float(row[5].split(';')[0])  # exclude the ending ';'
            data.append((user, label, x, y, z))
        except:
            # print('bad data row: ',row)
            pass


    data = np.asarray(data)

    # use sklearn.preprocessing.MinMaxScaler to normalize data into (0,1)
    xyz = data[:, 2:]
    scaler = MinMaxScaler()
    scaler.fit(xyz)
    data[:, 2:] = scaler.transform(xyz)

    # for a strict data split, we either use the same user's data in training or test
    data_train = data[data[:, 0] <= 30]  # user_id <= 30 for training, 80%~ish
    data_test = data[data[:, 0] > 30]  # the rest as test, 20%~ish

    WINDOW = 100
    STEP = 40
    x_train, y_train = create_segments_and_labels(data_train, WINDOW, STEP)
    x_test, y_test = create_segments_and_labels(data_test, WINDOW, STEP)

    ### make the label list binary with only chosenType being 1 and all other types being 0
    if chosenType != -1:
        y_test = (y_test == chosenType).astype(int)
        y_train = (y_train == chosenType).astype(int)

    return x_train, x_test, y_train, y_test
    # return x_train[:100], x_test[:100], y_train[:100], y_test[:100]  # for debug


def get_model():
    '''
    define the multi-task learning model
    :return:
    '''

    ################  block-0

    input_0 = Input(shape=(dim_x, dim_y, dim_z), name='input_0')

    conv_0 = Conv2D(8, kernel_size=(3, 3), activation="relu", name='conv_0')(input_0)

    ################  block-1

    conv_1_1_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_1_1')(conv_0)
    conv_1_1_2 = Conv2D(64, kernel_size=(2, 3), activation="relu", name='conv_1_1_2')(conv_1_1_1)


    conv_1_2_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_2_1')(conv_0)
    conv_1_2_2 = Conv2D(64, kernel_size=(2, 3), activation="relu", name='conv_1_2_2')(conv_1_2_1)

    ################  block-2

    flatten_2_1 = Flatten(name='flatten_2_1')(conv_1_1_2)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1 = Dense(40, activation="relu", name='dense_2_1')(drop_2_1)

    flatten_2_2 = Flatten(name='flatten_2_2')(conv_1_2_2)
    drop_2_2 = Dropout(rate=0.5, name='drop_2_2')(flatten_2_2)
    dense_2_2 = Dense(40, activation="relu", name='dense_2_2')(drop_2_2)

    flatten_2_3 = Flatten(name='flatten_2_3')(conv_1_2_2)
    drop_2_3 = Dropout(rate=0.5, name='drop_2_3')(flatten_2_3)
    dense_2_3 = Dense(40, activation="relu", name='dense_2_3')(drop_2_3)

    ################ block-3

    dense_3_1 = Dense(2, activation="softmax", name='dense_3_1')(dense_2_1)

    dense_3_3 = Dense(2, activation="softmax", name='dense_3_3')(dense_2_2)

    dense_3_0 = Dense(2, activation="softmax", name='dense_3_0')(dense_2_3)
    dense_3_2 = Dense(2, activation="softmax", name='dense_3_2')(dense_2_3)
    dense_3_4 = Dense(2, activation="softmax", name='dense_3_4')(dense_2_3)
    dense_3_5 = Dense(2, activation="softmax", name='dense_3_5')(dense_2_3)


    ###############

    model = keras.models.Model(input_0, [dense_3_0,
                                         dense_3_1,
                                         dense_3_2,
                                         dense_3_3,
                                         dense_3_4,
                                         dense_3_5])
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


dataset = 'hhar'
dim_x, dim_y, dim_z = 10, 10, 3 # dimension for corresponding dataset
n_task = 6



save_path = './checkpoint/{}/'.format(dataset)

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
        path = '/'.join(save_path.split('/') + ['best'])
        model.save(path)
