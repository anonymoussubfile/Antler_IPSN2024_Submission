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
    gsc dataset: it takes some time to calculate features,
    so we preprocess it and save time when you have to run many times
    '''


    '''
    download gsc dataset: 
    $ wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    we only use the following ten classes
    ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    '''

    # audio files in gsc have varying length, we pad or truncate them all to a fixed length
    pad = lambda a, l: a[0:l] if a.shape[0] > l else np.hstack((a, np.zeros(l - a.shape[0])))

    # parameters
    path = '../dataset/gsc/'
    N_FFT = 1024
    HOP_LENGTH = 1400
    N_MFCC = 32

    classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    features = []
    for classID, className in enumerate(classes):

        filenames = os.listdir(path + className)
        for i, filename in enumerate(filenames):

            audio, RATE = librosa.load(path + className + '/' + filename)
            audio = pad(audio, 22050)
            feature = librosa.feature.mfcc(y=audio, sr=RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
            feature = feature.reshape((N_MFCC, feature.shape[1] // dim_z, dim_z))

            features.append((feature, classID))

            if i % 500 == 0:
                print('{} {}: {}/{} processed'.format(classID, className, i, len(filenames)))

    PreprocessFeatures = features
    filename = 'PreprocessFeatures.npy'
    np.save(path + filename, PreprocessFeatures)

    print('\nGSC feature preprocessed and saved!\n\n\n')



def dataload_dataset(chosenType=-1):
    '''
    divide the original dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
                    -1: load the original dataset that has 10-class
    :return: training and testing dataset
    '''

    # load pre-computed features
    path = '../dataset/gsc/'
    filename = 'PreprocessFeatures.npy'
    data = np.load(path + filename, allow_pickle=True)

    features = defaultdict(list)
    for feature, classID in data:
        features[classID].append(feature)

    x_train, y_train, x_test, y_test = [], [], [], []
    for classID in features.keys():
        threshold = int(0.8 * len(features[classID]))  # use 80% for training, 20% for testing
        for i, feature in enumerate(features[classID]):
            if i > threshold:
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

    conv_0 = Conv2D(16, kernel_size=(12, 6), activation="relu", name='conv_0')(input_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='pool_0')(conv_0)

    ################  block-1

    conv_1_1 = Conv2D(32, kernel_size=(6, 3), activation="relu", name='conv_1_1')(pool_0)
    pool_1_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1')(conv_1_1)
    flatten_1_1 = Flatten(name='flatten_1_1')(pool_1_1)
    drop_1_1 = Dropout(rate=0.5, name='drop_1_1')(flatten_1_1)
    dense_1_2 = Dense(280, activation="relu", name='dense_1_2')(drop_1_1)

    ################  block-2

    dense_2_1 = Dense(128, activation="relu", name='dense_2_1')(dense_1_2)

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

dataset = 'gsc'
dim_x, dim_y, dim_z = 32, 16, 1   # input sample dimension for corresponding dataset


# please run the following feature_extraction() first if you are running this for the first time
# it pre-computes necessary features that we need to reload repeatedly
# once pre-computed, you can comment this line
# feature_extraction()
# exit()

history_acc1 = []
for multiple_run in range(1,5):

    save_path = './checkpoint/{}/run{}/'.format(dataset, multiple_run)

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

                print('\n\n\n',history_acc1,'\n\n\n')



            cnt += 1
            if cnt >= 10 and epoch > 30:  # if no higher acc achieved within 10 consecutive epochs, we exit
                break

    _, x_test, _, y_test = dataload_dataset()
    acc1 = evaluate1(10, save_path, x_test, y_test)
    history_acc1.append((multiple_run, '{:.4f}'.format(acc1)))

print('\n\n\n',history_acc1,'\n\n\n')
