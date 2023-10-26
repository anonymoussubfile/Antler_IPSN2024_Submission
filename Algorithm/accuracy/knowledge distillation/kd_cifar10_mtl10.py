'''
explanation of the name of the file
kd: knowledge distillation
mtl10: multi-task learning, task number is 10
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout


class Distiller(keras.Model):
    def __init__(self, student, teacher, n_task):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.n_task = n_task

    def compile(
            self,
            optimizer,
            metrics,
            distillation_loss_fn,
            temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature

    def train_step(self, data):

        def crop10to2(teacher_softmax):
            '''
            our teacher model is a 10-class classification model
            our student model is a binary classification model
            we have to crop the softmax of teacher model from 10-dim to 2-dim
            our student model is a multi-task learning model which has n binary classification models

            :param teacher_softmax: 10 probabilities for 10 classes
            :return: the ret should be a list of n binary output
                     each output is for one binary classification problem, [negative_prob, positive_prob]
            '''

            ret = []

            for i in range(self.n_task):
                positive = teacher_softmax[:, i]
                ret.append(tf.stack([1 - positive, positive], axis=1))

            return ret

        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # our teacher and student models should output probabilities (after softmax)
            distillation_loss = (
                    self.distillation_loss_fn(
                        crop10to2(teacher_predictions),
                        student_predictions,
                    )
                    * self.temperature ** 2
            )

            # we only use distillation loss as we do not have access to the original training set
            loss = distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"dl_loss": distillation_loss})
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}

        ave_acc = tf.reduce_mean([m.result() for m in self.metrics])  # averaged accuracy over all classes
        results.update({"ave_acc": ave_acc})
        return results

    def evaluate1(self, x_test, y_test):
        # measure the accuracy of the student model differently
        # we select the class which has the highest probability as the prediction result
        # compare it with the test set
        # if they match, we consider this data point is correctly classified

        pre = self.student.predict(x_test)
        pre = np.asarray(pre)

        n_class, n_sample, n_output = pre.shape
        cnt = 0
        for i in range(n_sample):
            cnt += 1 if pre[:, i, 1].argmax() == y_test[i][0] else 0
        acc = cnt / n_sample

        print(acc)
        return acc

    def evaluate2(self, x_test, y_test):
        # measure the accuracy of the student model differently
        # we check the output of all tasks individually
        # only when there is only one class saying Yes and all others saying No
        # plus, the class saying Y must match with the true class
        # then, we consider this data point is correctly classified

        pre = self.student.predict(x_test)
        pre = np.asarray(pre)

        n_class, n_sample, n_output = pre.shape
        cnt = 0
        for i in range(n_sample):
            yes = pre[:, i, 1] > pre[:, i, 0]
            cnt += 1 if np.sum(yes) == 1 and yes.argmax() == y_test[i][0] else 0
        acc = cnt / n_sample

        print(acc)
        return acc


def dataload_mtl():
    '''
    :return: data for training MTL models
    '''
    trainX = dataload_cifar10(0)[0]
    testX = dataload_cifar10(0)[1]

    trainY = []
    testY = []
    for i in range(10):
        trainY.append(dataload_cifar10(i)[2])
        testY.append(dataload_cifar10(i)[3])

    return trainX, trainY, testX, testY


def dataload_cifar10(chosenType=-1):
    '''
    divide the original dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
                    -1: load the original dataset that has 10-class
    :return: training and testing dataset
    '''

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

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
    # return x_train[:1000], x_test[:1000], y_train[:1000], y_test[:1000]


def get_model():
    '''
    define the multi-task learning model
    :return:
    '''

    ################  block-0

    input_0 = Input(shape=(dim_x, dim_y, dim_z), name='input_0')

    conv_0 = Conv2D(16, kernel_size=(3, 3), activation="relu", name='conv_0')(input_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='pool_0')(conv_0)

    ################  block-1

    conv_1_1 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_1')(pool_0)
    pool_1_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1_1')(conv_1_1)

    conv_1_2 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_2')(pool_0)
    pool_1_2 = MaxPooling2D(pool_size=(2, 2), name='pool_1_2')(conv_1_2)

    conv_1_3 = Conv2D(32, kernel_size=(3, 3), activation="relu", name='conv_1_3')(pool_0)
    pool_1_3 = MaxPooling2D(pool_size=(2, 2), name='pool_1_3')(conv_1_3)

    ################  block-2

    conv_2_1 = Conv2D(52, kernel_size=(3, 3), activation="relu", name='conv_2_1')(pool_1_1)
    pool_2_1 = MaxPooling2D(pool_size=(2, 2), name='pool_2_1')(conv_2_1)
    flatten_2_1 = Flatten(name='flatten_2_1')(pool_2_1)
    drop_2_1 = Dropout(rate=0.5, name='drop_2_1')(flatten_2_1)
    dense_2_1_1 = Dense(128, activation="relu", name='dense_2_1_1')(drop_2_1)
    dense_2_1_2 = Dense(128, activation="relu", name='dense_2_1_2')(dense_2_1_1)

    conv_2_2 = Conv2D(52, kernel_size=(3, 3), activation="relu", name='conv_2_2')(pool_1_2)
    pool_2_2 = MaxPooling2D(pool_size=(2, 2), name='pool_2_2')(conv_2_2)
    flatten_2_2 = Flatten(name='flatten_2_2')(pool_2_2)
    drop_2_2 = Dropout(rate=0.5, name='drop_2_2')(flatten_2_2)
    dense_2_2_1 = Dense(128, activation="relu", name='dense_2_2_1')(drop_2_2)
    dense_2_2_2 = Dense(128, activation="relu", name='dense_2_2_2')(dense_2_2_1)

    conv_2_3 = Conv2D(52, kernel_size=(3, 3), activation="relu", name='conv_2_3')(pool_1_2)
    pool_2_3 = MaxPooling2D(pool_size=(2, 2), name='pool_2_3')(conv_2_3)
    flatten_2_3 = Flatten(name='flatten_2_3')(pool_2_3)
    drop_2_3 = Dropout(rate=0.5, name='drop_2_3')(flatten_2_3)
    dense_2_3_1 = Dense(128, activation="relu", name='dense_2_3_1')(drop_2_3)
    dense_2_3_2 = Dense(128, activation="relu", name='dense_2_3_2')(dense_2_3_1)

    conv_2_4 = Conv2D(52, kernel_size=(3, 3), activation="relu", name='conv_2_4')(pool_1_3)
    pool_2_4 = MaxPooling2D(pool_size=(2, 2), name='pool_2_4')(conv_2_4)
    flatten_2_4 = Flatten(name='flatten_2_4')(pool_2_4)
    drop_2_4 = Dropout(rate=0.5, name='drop_2_4')(flatten_2_4)
    dense_2_4_1 = Dense(128, activation="relu", name='dense_2_4_1')(drop_2_4)
    dense_2_4_2 = Dense(128, activation="relu", name='dense_2_4_2')(dense_2_4_1)

    ################ block-3

    dense_3_1 = Dense(2, activation="softmax", name='dense_3_1')(dense_2_1_2)

    dense_3_2 = Dense(2, activation="softmax", name='dense_3_2')(dense_2_2_2)
    dense_3_3 = Dense(2, activation="softmax", name='dense_3_3')(dense_2_2_2)
    dense_3_4 = Dense(2, activation="softmax", name='dense_3_4')(dense_2_2_2)
    dense_3_5 = Dense(2, activation="softmax", name='dense_3_5')(dense_2_2_2)
    dense_3_6 = Dense(2, activation="softmax", name='dense_3_6')(dense_2_2_2)
    dense_3_7 = Dense(2, activation="softmax", name='dense_3_7')(dense_2_2_2)
    dense_3_8 = Dense(2, activation="softmax", name='dense_3_8')(dense_2_2_2)

    dense_3_9 = Dense(2, activation="softmax", name='dense_3_9')(dense_2_3_2)

    dense_3_10 = Dense(2, activation="softmax", name='dense_3_10')(dense_2_4_2)

    ###############

    model = tf.keras.models.Model(input_0, [dense_3_1,
                                            dense_3_2,
                                            dense_3_3,
                                            dense_3_4,
                                            dense_3_5,
                                            dense_3_6,
                                            dense_3_7,
                                            dense_3_8,
                                            dense_3_9,
                                            dense_3_10])
    return model


####-----------------------------------------------------------##
dataset = 'Vgg16'
dim_x, dim_y, dim_z = 32, 32, 3  # CIFAR10

student = get_model()
teacher = tf.keras.models.load_model('./checkpoint/cifar10/softmax/{}/best'.format(dataset))

# evaluate the pre-trained teacher model
# _, x_test, _, y_test = dataload_cifar10()
# teacher.evaluate(x_test, y_test)
# exit()

x_train, x_test, y_train, y_test = dataload_cifar10()  # for 10-class classification
trainX, trainY, testX, testY = dataload_mtl()  # for binary classification

student_save_path = './checkpoint_kd/cifar10/{}'.format(dataset)

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher, n_task=10)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=['acc'],
    distillation_loss_fn=keras.losses.KLDivergence(),
    temperature=3,
)

# load pretrained student model
distiller.student = models.load_model('./checkpoint_kd/cifar10/Vgg16/ave_acc_0.9515')

# Distill teacher to student
epochs = 100
best_ave_acc = 0.94
best_acc1 = 0.7
for i in range(epochs):
    history = distiller.fit(trainX, trainY, epochs=1, validation_data=(testX, testY))
    ave_acc = float('{:.4f}'.format(history.history['val_ave_acc'][0]))
    acc1 = distiller.evaluate1(x_test, y_test)
    acc2 = distiller.evaluate2(x_test, y_test)
    print('\n\n\n{}/{} done\nave_acc={}\nacc1={}\nacc2={}\n'.format(i + 1, epochs, ave_acc, acc1, acc2))
    if acc1 > best_acc1:
        best_ave_acc = ave_acc
        best_acc1 = acc1

        save_path = '/'.join(student_save_path.split('/') + ['acc1_{}_acc2_{}_ave_acc_{}'.format(acc1, acc2, ave_acc)])
        distiller.student.save(save_path)

# Evaluate student on test dataset
distiller.evaluate(testX, testY)

distiller.evaluate1(x_test, y_test)
distiller.evaluate2(x_test, y_test)

