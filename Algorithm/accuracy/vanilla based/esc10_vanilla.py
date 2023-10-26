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
import librosa, random

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

    ################  block-2

    conv_2_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_2_1')(pool_1_1)
    pool_2_1 = MaxPooling2D(pool_size=(2, 2), name='pool_2_1')(conv_2_1)
    flatten_2_1 = Flatten(name='flatten_2_1')(pool_2_1)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1_1 = Dense(128, activation="relu", name='dense_2_1_1')(drop_2_1)
    dense_2_1_2 = Dense(64, activation="relu", name='dense_2_1_2')(dense_2_1_1)

    ################ block-3

    dense_3_1 = Dense(2, activation="softmax", name='dense_3_1')(dense_2_1_2)

    ###############

    model = keras.models.Model(input_0, dense_3_1)
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


save_path = './checkpoint/esc10/'
dim_x, dim_y, dim_z = 32, 108, 2   # input sample dimension for corresponding dataset



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
            best_acc = acc
            path = '/'.join(save_path.split('/') + ['type_{}_val_acc_{}'.format(type, acc)])
            model.save(path)
            model.save(path_best)
            cnt = 0

        cnt += 1
        if cnt >= 10:  # if no higher acc achieved within 10 consecutive epochs, we exit
            break

_, x_test, _, y_test = dataload_dataset()
evaluate1(10, save_path, x_test, y_test)

