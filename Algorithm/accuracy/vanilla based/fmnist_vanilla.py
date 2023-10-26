'''
explanation of the name of the file
kd: knowledge distillation
mtl10: multi-task learning, task number is 10
'''


import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import os, shutil

def dataload_cifar10(chosenType=-1):
    '''
    divide the original dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
                    -1: load the original dataset that has 10-class
    :return: training and testing dataset
    '''

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    ### Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # if MNIST, we need to reshape
    x_train = np.reshape(x_train, (-1, dim_x, dim_y, dim_z))
    x_test = np.reshape(x_test, (-1, dim_x, dim_y, dim_z))

    ### make the label list binary with only chosenType being 1 and all other types being 0
    if chosenType != -1:
        y_test = (y_test == chosenType).astype(int)
        y_train = (y_train == chosenType).astype(int)

    return x_train, x_test, y_train, y_test
    # return x_train[:1000], x_test[:1000], y_train[:1000], y_test[:1000]  # for debug


def get_model():
    '''
    define the multi-task learning model for cifar10
    :return:
    '''

    ################  block-0

    input_0 = Input(shape=(dim_x, dim_y, dim_z), name='input_0')

    conv_0 = Conv2D(24, kernel_size=(3, 3), activation="relu", name='conv_0')(input_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='pool_0')(conv_0)

    ################  block-1

    conv_1_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_1')(pool_0)
    pool_1_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1')(conv_1_1)

    ################  block-2

    conv_2_1 = Conv2D(52, kernel_size=(3, 3), activation="relu", name='conv_2_1')(pool_1_1)
    pool_2_1 = MaxPooling2D(pool_size=(2, 2), name='pool_2_1')(conv_2_1)
    flatten_2_1 = Flatten(name='flatten_2_1')(pool_2_1)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1_1 = Dense(256, activation="relu", name='dense_2_1_1')(drop_2_1)
    dense_2_1_2 = Dense(128, activation="relu", name='dense_2_1_2')(dense_2_1_1)

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


save_path = './checkpoint/fmnist/'
dim_x, dim_y, dim_z = 28, 28, 1   # input sample dimension for corresponding dataset



for type in range(10):

    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['acc'])

    epochs = 50
    best_acc = 0  # only save model if acc is above this one
    cnt = 0

    path_best = '/'.join(save_path.split('/') + ['type_{}'.format(type)])

    x_train, x_test, y_train, y_test = dataload_cifar10(type)
    for epoch in range(epochs):

        print('\ntype={}, epoch={}/{}'.format(type, epoch, epochs))
        hist = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
        acc = float('{:.4f}'.format(hist.history['val_acc'][0]))


        if acc > best_acc:  # only save the model if it has better targeted accuracy

            # delete folder with lower accuracy
            all = os.listdir(save_path)
            for file_or_dir in all:
                if 'type_{}_'.format(type) in file_or_dir:
                    acc_exist = float(file_or_dir.split('_')[-1])
                    if acc_exist < acc: # delete folder with lower accuracy
                        shutil.rmtree(save_path + file_or_dir, ignore_errors=True)

            # save the best accuracy model
            best_acc = acc
            path = '/'.join(save_path.split('/') + ['type_{}_val_acc_{}'.format(type, acc)])
            model.save(path)
            model.save(path_best)
            cnt = 0

        cnt += 1
        if cnt >= 10:  # if no higher acc achieved within 10 consecutive epochs, we exit
            break

_, x_test, _, y_test = dataload_cifar10()
evaluate1(10, save_path, x_test, y_test)

