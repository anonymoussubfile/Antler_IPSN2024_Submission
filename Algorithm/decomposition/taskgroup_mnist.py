import tensorflow as tf

print('tensorflow version:', tf.__version__)  # we use tensorflow 2.x

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import backend as F
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten

from scipy import stats
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import itertools, more_itertools
import time

import xlsxwriter
import random

##################################################################################################################
# helper functions

def get_possible_branch_locations():
    '''
    our model may have different types of layers, e.g., Conv, Dense, MaxPooling, Flatten
    we will attach MaxPooling, Flatten layers to the closest Conv layer that is before them
    because MaxPooling Flatten layers do not have weight parameters
    Besides, we do not branch out after the final layer
    :return : the index of layers after which we can be branch out
    Example, model = Conv(0) + Conv(1) + Conv(2) + MaxPooling(3) + Flatten(4) + Dense(5) + Dense(6) + Dense(7)
    we will return [0, 1, 4, 5, 6]
    '''

    model = model_train(train=False)

    list = []
    for idx, layer in enumerate(model.layers):

        if idx == len(model.layers) - 1:  # we do not branch out on the final layer
            continue

        if 'input' in layer.name:  # we do not branch out on input layer
            continue

        # # we assume layers (e.g. MaxPooling, Flatten) other than Conv and Dense will only exist after Conv layers
        # # and Dense layers are all consecutively connected to each other
        if ('conv' in model.layers[idx].name and 'conv' in model.layers[idx + 1].name) or \
                ('conv' not in model.layers[idx].name and 'conv' in model.layers[idx + 1].name) or \
                ('conv' not in model.layers[idx].name and 'dense' in model.layers[idx + 1].name) or \
                ('dense' in model.layers[idx].name):
            list.append(idx)

    return list


def get_segment_byBlock(data, Idx=[0, 1, 4]):
    '''
    segment data (layer-wise network overhead savings) into four parts by the three branch out points
    :param data: layer-wise network overhead savings, e.g., weight-reloading savings, computational savings
    :param Idx: the index of branch out points w.r.t all possible branchable points, For example, if len(RSM) = D, then
            it means there are D branchable points, Idx is corresponding to range(D)
    :return: segmented result
    '''

    # Idx = branchIdx_inRSM  # use a shorter name
    segmented = [sum(data[:Idx[0] + 1]),  # block_0 - always shared by all tasks
                 sum(data[Idx[0] + 1:Idx[1] + 1]),  # block_1
                 sum(data[Idx[1] + 1:Idx[2] + 1]),  # block_2
                 sum(data[Idx[2] + 1:])]  # block_3

    return segmented


def get_weightsize_byBlock(Idx):
    '''
    get block-wise weight size - in number of parameters
    '''

    model = model_train(train=False)

    var = model.trainable_variables
    sizeByLayer = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]

    sizeByBlock = get_segment_byBlock(sizeByLayer, Idx=Idx)

    return sizeByBlock


def get_ComputationalSavings(Idx):
    '''
    get block-wise computational savings w.r.t time
    '''

    # # layer-wise inference time array, this array comes from our experiment results
    timesavingsByLayer = [0.74, 3.81, 1.18, 0.073, 0.05, 0.008]  # MNIST
    # timesavingsByLayer = [2.57, 5.25, 1.85, 0.048, 0.042, 0.009]  # CIFAR10
    # timesavingsByLayer = [2.57, 5.25, 1.85, 0.048, 0.042, 0.009]  # SVHN
    # timesavingsByLayer = [2.6,  5.28, 1.86, 0.048, 0.041, 0.004]  # GTSRB
    # timesavingsByLayer = [1.3,  3.01, 0.014, 0.017, 0.006]  # GSC


    timesavingsByBlock = get_segment_byBlock(timesavingsByLayer, Idx=Idx)

    return timesavingsByBlock


def ascending(array):
    '''
    make array strictly ascending
    '''
    for i in range(1, len(array)):
        if array[i] < array[i - 1]:
            array[i] = array[i - 1]
    return array


def descending(array):
    '''
    make array strictly descending
    '''
    for i in range(1, len(array)):
        if array[i] > array[i - 1]:
            array[i] = array[i - 1]
    return array


##################################################################################################################


def dataload_mnist(chosenType):
    '''
    divide the original MNIST dataset into single digit recognition dataset (a binary classification problem)
    with the chosenType = 1, all other types = 0
    :param chosenType: the class type you pick as the 1 (other types will be 0)
    :return: training and testing dataset
    '''

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    ### Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    ### make the label list binary with only chosenType being 1 and all other types being 0
    y_test = [1 if c == chosenType else 0 for c in y_test]
    y_train = [1 if c == chosenType else 0 for c in y_train]

    ### expand dimension
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    y_train = tf.expand_dims(y_train, -1)
    y_test = tf.expand_dims(y_test, -1)

    return x_train, x_test, y_train, y_test


def model_train(train=True, chosenType=0):
    '''
    Retrain a new model or load a pretrained model
    :param train: True means to retrain a model; False means to load a pretrained model
    :param chosenType: the class type you use as 1
    :return: return the keras model
    '''

    if not train:  # load model data from pretrained model
        model = keras.models.load_model('pretrained/mnist/mnist_{}'.format(chosenType))

    else:  # else build and train a new model, and save it

        x_train, x_test, y_train, y_test = dataload_mnist(chosenType)

        # # implement the same network architecture from NeuralWeightVirtualization for MNIST dataset
        model = models.Sequential()
        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(10))

        # model.summary()

        # compile and train
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

        # save model
        model.save('pretrained/mnist/mnist_{}'.format(chosenType))
        print('Model saved: pretrained/mnist/mnist_{}'.format(chosenType))

    return model


def RDM_Calc(K, chosenType=0, train=False, getFuncNumber=False):
    '''
    calculate the Representation Dissimilarity Matrix (RDM) for each possible branch out point
    this function is for one task, you need to run this function repeatedly for all tasks
    :param K: the number of images to use for calculation, better to use 50 or more
    :param chosenType: the class type you use as 1 for this task
    :param getFuncNumber: if this param is True, we just return the number of func, a.k.a the number of branch out points
    :return: RDM for this task
    '''

    model = model_train(chosenType=chosenType, train=train)
    x_train, x_test, _, _ = dataload_mnist(chosenType)
    images = x_test[:K]

    # # get the name of each layers for debug
    # layersName = {idx: [layer.name, layer.output.shape] for idx, layer in enumerate(model.layers)}

    '''
    set up functions to read the output of every intermediate layer 
    we only read intermediate results after layers with trainable weights
    for layers without weights, e.g., MaxPooling, Flatten, we attach them to the closest conv layer before it
    as for how to get those layers, it is implemented in get_possible_branch_locations()
    '''
    funcList = []
    possible_branchLoc = get_possible_branch_locations()
    for idx in possible_branchLoc:
        funcList.append(F.function([model.layers[0].input], [model.layers[idx].output]))

    if getFuncNumber:
        return len(funcList)

    # # read outputs from intermediate layers of the model using K images
    K = len(images)  # K images were used
    outs = []
    for func in funcList:
        out = func(images)[0]

        # # the 3-D tensors are linearized to 1-D tensors
        outs.append(out.reshape(out.shape[0], int(out.size / out.shape[0])))

    # # after we get intermediate results, we use them to calculate the Representation Dissimilarity Matrix (RDM)
    RDM = np.zeros((len(outs), K, K))
    for idx, out in enumerate(outs):
        # # for each func (a.k.a each branch out point)
        for i in range(K):
            for j in range(i, K):
                # # pearson correlation coefficient tells the correlation, we need the dissimilarity, so we use (1 - p)
                RDM[idx][i][j] = 1 - stats.pearsonr(out[i], out[j])[0]
                RDM[idx][j][i] = RDM[idx][i][j]  # the matrix is symmetric

    return RDM


def RSM_Calc(K):
    '''
    calculate task-wise Representation Similarity Matrix (RSM) - the paper calls it task affinity tensor A
    :param K: the number of images to use for RDM calculation
    :return:
    '''

    print('calculate task-wise Representation Similarity Matrix (RSM)...')

    # # some parameters
    T = 10  # number of tasks, mnist dataset has 10 classes and we divide it into 10 individual tasks
    D = RDM_Calc(1, getFuncNumber=True)  # D is the number of branch out points
    RSM = np.zeros((D, T, T))

    # # calculate RDM for each task
    # RDM = [RDM_Calc(K, chosenType=t) for t in range(T)]
    RDM = []
    for t in range(T):
        RDM.append(RDM_Calc(K, chosenType=t))
        print('RDM: {}/{} task done.'.format(t + 1, T))

    # #
    for d in range(D):

        for i in range(T):
            for j in range(T):
                # # extract RDM of the d_th division point for task_i and task_j
                m1, m2 = RDM[i][d], RDM[j][d]

                # # extract the upper triangle of the matrix and flatten them into a list
                p1 = [elem for ii, row in enumerate(m1) for jj, elem in enumerate(row) if ii < jj]
                p2 = [elem for ii, row in enumerate(m2) for jj, elem in enumerate(row) if ii < jj]

                # # calculate the Spearman’s correlation coefficient for task_i and task_j at the d_th division point
                RSM[d][i][j] = stats.spearmanr(p1, p2).correlation

    print('RSM done...')
    return RSM


def partition(collection):
    '''
    Bell Number reference: https://en.m.wikipedia.org/wiki/Bell_number
    implementation reference: https://stackoverflow.com/questions/19368375/set-partitions-in-python
    return all possible partitions of a list of unique numbers
    "A partition of a set X is a set of non-empty subsets of X such that
    every element x in X is in exactly one of these subsets"
    :param collection: a list of unique numbers
    :return: all possible partitions
    '''
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # # put `first` in its own subset
        yield [[first]] + smaller


def ScoreCalc(clusters, depth, RSM):
    '''
    calculate the task similarity/affinity score for a given division point
    :param clusters: the input cluster list, e.g. for [[1,2],[3,4]], [1,2] is a cluster, [3,4] is a cluster
    :param depth: the division point
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :return: use 1 to minus the averaged max distance between the dissimilarity scores of the elements in every cluster
    '''

    tobeAveraged = 0
    for cluster in clusters:
        n = len(cluster)

        if n <= 1:
            tobeAveraged += 0  # the dissimilarity between a task and itself is 0
        else:
            '''
            calculate all pair-wise dissimilarity scores for all elements in this cluster
            then only keep the largest score and add it to the tobeAveraged variable
            '''
            max_score = float('-inf')
            for i in range(n - 1):
                for j in range(i + 1, n):

                    # # 1 - RSM(d, i, j) is the dissimilarity score between task_i and task_j at division point d
                    if 1 - RSM[depth][cluster[i]][cluster[j]] > max_score:
                        max_score = 1 - RSM[depth][cluster[i]][cluster[j]]

            tobeAveraged += max_score

    # # on Apr 2022, we decided to use similarity score (the higher the better) to evaluate a decomposition tree
    # # so, we now use 1 - tobeAveraged / len(clusters), instead of tobeAveraged / len(clusters)
    return 1 - tobeAveraged / len(clusters)


def weak_compositions(boxes, balls, parent=tuple(), depth=0, constraint=[float('inf')] * 30):
    '''
    when we branch out from layer L to layer L + 1, for the partition at layer L, it is easy
    you can just use more_itertools.set_partitions which allows you to specify the number of partitions you want
    however, for layer L + 1, it is not trivial to do, let's use an example to analysis

    Example: assume at layer L, the partition result is [[1,2],[3],[4,5,6]] and we have 4 partitions at layer L + 1
        so at layer L, we have 3 partitions in [[1,2],[3],[4,5,6]], in order to branch into a 4-partition result at layer L + 1
        we can divide the [1,2] so we have [[1],[2],[3],[4,5,6]] at layer L + 1, or we can divide [4,5,6] and have [[1,2],[3],[4],[5,6]]
        and there are many other ways to divide [4,5,6] into two parts.

        In our solution, we can covert this into a "toss N balls into M boxes" problem
        the number of balls (N) is the number of partitions at layer L + 1
        the number of boxes (M) is the number of partitions at layer L
        in our example, N = 4, M = 3
        then valid tossing strategy includes: [4, 0, 0], [3, 1, 0], [1, 2, 1] ...
        values in each tossing strategy represents that number of parts we should divide the corresponding partition in layer L
        e.g. [1, 2, 1] means that
                            we divide the 1st part - [1,2] of layer L into 1 part
                            we divide the 2nd part - [3] of layer L into 2 parts  (does not make sense)
                            we divide the 3rd part - [4,5,6] of layer L into 1 part (does not make sense)
        As you can see here, the strategy may have have non-sense for some divisions. We need to add constraints
        for the general "toss N balls into M boxes" problem
        Constraints: (1) each box must have at least one ball - here means each part of layer L if not divided should be at least kept as it is
                     (2) the number of balls in a box should be smaller than the number of values in its corresponding part at layer L
                            e.g. you can not divide [1,2] into three parts

        Final solution, we borrow the function wrote in this post https://stackoverflow.com/questions/4647120/next-composition-of-n-into-k-parts-does-anyone-have-a-working-algorithm
            go to the above link and ctr+F to search "weak_compositions" and you will find the code we borrowed
            we modify the code by eliminating (1) any 0 value, and (2) any value that exceed its corresponding constraint value

    :param depth: depth of the recursion, we use this to index the value of constraint for later comparsion
    :param constraint: the number of values in each part of layer L
                        e.g. [[1,2],[3],[4,5,6]] at layer L, the constraint should be [2,1,3]
                       the length of constraint should be larger than the number of parts at layer L, 30 should be big enough for our case

    '''

    max_depth = len(constraint)

    if boxes > 1:
        for i in range(1, balls):  # modified the original range(balls + 1), to eliminate 0 value

            if balls - i > constraint[depth]:  # apply constraint for all non-final depth in recursion
                continue

            if depth == max_depth - 2 and i > constraint[-1]:  # apply constraint for the final depth in recursion
                continue

            for x in weak_compositions(boxes - 1, i, parent + (balls - i,), depth + 1, constraint=constraint):
                yield x
    else:
        yield parent + (balls,)


def clustering_withBudget(RSM, N=5, Budget=5):
    '''
    enumerate all possible trees under a budget
    we only enumerate trees that use the maximum possible budget which is no greater than the given input Budget
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :param N: total number of tasks
    :param Budget: the number of nodes on top of the minimal budget
    '''

    # N = 9 # number of tasks
    tasks = list(range(N))
    queue = []  # to save all possible trees, and their score and model size

    # get the weight size of each layer
    model = model_train(train=False)
    var = model.trainable_variables
    sizeByLayer = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]
    sizeByBlock = [sizeByLayer[0] + sizeByLayer[1]] + sizeByLayer[2:]

    '''
    our structure only has 4 layers of blocks:
        0th layer: only 1 node
        1st layer: flexible, to be determined
        2nd layer: flexible, to be determined
        3th layer: len(task) nodes because you must branch into individual task at this layer
    for a given budget, we enumerate all possible arrangements for the 1st & 2nd layers

    Constraints: 
        the budget nodes of the 1st layer should be no greater than that of the 2nd layer
        the budget nodes of all layers should be less than the number of tasks
    '''
    arrangement = []
    for i in range(1, int(Budget / 2) + 1):
        if i <= N and Budget - i <= N:
            arrangement.append([i, Budget - i])

    cnt1 = cnt2 = 0
    for arrange in arrangement:
        for clusters0 in more_itertools.set_partitions(tasks,
                                                       k=arrange[0]):  # arrange[0]: the number of nodes for 1st layer

            ''' Variable Name Explanation 
            clustersX, depthX, scoreX: 'X' means the Xth branch out point
            clustersX: means the branched-out result at Xth branch out point
            scoreX: means the dissimilarity score for clustersX
            '''

            # continue

            depth0, depth1 = 0, 1
            score0 = ScoreCalc(clusters0, depth0, RSM)

            # print(clusters0)
            # print('--->')

            for decompose in weak_compositions(arrange[0], arrange[1], constraint=[len(p) for p in clusters0]):

                # for clusters1 in itertools.product(*[partition(cluster) for cluster in clusters0]):
                for clusters1 in itertools.product(
                        *[more_itertools.set_partitions(cluster, k=k) for cluster, k in zip(clusters0, decompose)]):

                    '''
                    we use itertools.product to output all combinations of possible subtrees
                    please refer to: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
                    to understand how itertools.product(*[ ]) works
                    '''

                    # strip the out-layer bracket
                    res = []
                    for l in clusters1:
                        res += l
                    clusters1 = res
                    # print(res)

                    if len(clusters1) != arrange[1]:
                        cnt1 += 1  # unnecessary searches
                        continue

                    cnt2 += 1
                    score1 = ScoreCalc(clusters1, depth1, RSM)

                    '''
                    our paper only has three layers of branch out points, for the final one, it is always branched into single tasks
                    so, we do not need to continue the processing of clusters2/score2
                    clusters2 will all be single tasks, e.g. clusters2 should always be [[0],[1],[2],[3],[4]] if 5 tasks in total. 
                    then score2 will always be 0; 
                    '''

                    # summarize all info and pack them into the queue
                    score = score0 + score1  # total dissimilarity score of all branch out points, score2 is omitted as it is 0
                    # model_size = sizeByBlock[0] + len(clusters0) * sizeByBlock[1] \
                    #              + len(clusters1) * sizeByBlock[2] \
                    #              + len(tasks) * sizeByBlock[3]  # total model size this tree requires
                    model_size = sizeByBlock[0] + sizeByBlock[1] + sizeByBlock[2] + sizeByBlock[3]
                    tree = [clusters0, '--->', clusters1]  # decomposition details

                    queue.append([score, model_size, tree])

                # print('\n\n')

            queue.sort(key=lambda x: (x[1], x[0]))  # first sort by model_size, then by score
    print('cnt1 = {}, cnt2 = {}, unnecessary searches ratio = {:.3}'.format(cnt1, cnt2, cnt1 / (cnt1 + cnt2)))
    return queue


def clustering_withBudget_old(RSM, N=5, Budget=5):
    '''
    enumerate all possible trees under a budget
    we only enumerate trees that use the maximum possible budget which is no greater than the given input Budget
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :param N: total number of tasks
    :param Budget: the number of nodes on top of the minimal budget
    '''

    # N = 9 # number of tasks
    tasks = list(range(N))
    queue = []  # to save all possible trees, and their score and model size

    # get the weight size of each layer
    model = model_train(train=False)
    var = model.trainable_variables
    sizeByLayer = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]
    sizeByBlock = [sizeByLayer[0] + sizeByLayer[1]] + sizeByLayer[2:]

    '''
    our structure only has 4 layers of blocks:
        0th layer: only 1 node
        1st layer: flexible, to be determined
        2nd layer: flexible, to be determined
        3th layer: len(task) nodes because you must branch into individual task at this layer
    for a given budget, we enumerate all possible arrangements for the 1st & 2nd layers

    Constraints: 
        the budget nodes of the 1st layer should be no greater than that of the 2nd layer
        the budget nodes of all layers should be less than the number of tasks
    '''
    arrangement = []
    for i in range(1, int(Budget / 2) + 1):
        if i <= N and Budget - i <= N:
            arrangement.append([i, Budget - i])

    cnt1 = cnt2 = 0
    for arrange in arrangement:
        for clusters0 in more_itertools.set_partitions(tasks,
                                                       k=arrange[0]):  # arrange[0]: the number of nodes for 1st layer

            ''' Variable Name Explanation 
            clustersX, depthX, scoreX: 'X' means the Xth branch out point
            clustersX: means the branched-out result at Xth branch out point
            scoreX: means the dissimilarity score for clustersX
            '''

            # continue

            depth0, depth1 = 0, 1
            score0 = ScoreCalc(clusters0, depth0, RSM)

            # print(clusters0)
            # print('--->')

            for clusters1 in itertools.product(*[partition(cluster) for cluster in clusters0]):
                # for clusters1 in itertools.product(*[more_itertools.set_partitions(cluster, k = k) for cluster, k in zip(clusters0, decompose)]):

                '''
                we use itertools.product to output all combinations of possible subtrees
                please refer to: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
                to understand how itertools.product(*[ ]) works
                '''

                # strip the out-layer bracket
                res = []
                for l in clusters1:
                    res += l
                clusters1 = res
                # print(res)

                if len(clusters1) != arrange[1]:
                    cnt1 += 1  # unnecessary searches
                    continue

                cnt2 += 1
                score1 = ScoreCalc(clusters1, depth1, RSM)

                '''
                our paper only has three layers of branch out points, for the final one, it is always branched into single tasks
                so, we do not need to continue the processing of clusters2/score2
                clusters2 will all be single tasks, e.g. clusters2 should always be [[0],[1],[2],[3],[4]] if 5 tasks in total. 
                then score2 will always be 0; 
                '''

                # summarize all info and pack them into the queue
                score = score0 + score1  # total dissimilarity score of all branch out points, score2 is omitted as it is 0
                model_size = sizeByBlock[0] + len(clusters0) * sizeByBlock[1] \
                             + len(clusters1) * sizeByBlock[2] \
                             + len(tasks) * sizeByBlock[3]  # total model size this tree requires
                # model_size = sizeByBlock[0] + sizeByBlock[1] + sizeByBlock[2] + sizeByBlock[3]
                tree = [clusters0, clusters1]  # decomposition details

                queue.append([score, model_size, tree])

            # print('\n\n')

            queue.sort(key=lambda x: (x[1], x[0]))  # first sort by model_size, then by score
    print('cnt1 = {}, cnt2 = {}, unnecessary searches ratio = {:.3}'.format(cnt1, cnt2, cnt1 / (cnt1 + cnt2)))
    return queue


def GetBranchingInfo(branchIdx_inRSM=[0, 2, 4]):
    '''
    to be deleted
    :param branchIdx_inRSM: we have 3 branch out points, branchIdx has the index of the 3 branch out points w.r.t RSM
    :return:
    '''

    # get the weight size of each layer
    model = model_train(train=False)

    possible_branchLoc = get_possible_branch_locations()  # get the index of possible branch out locations
    branchIdx_inLayer = [possible_branchLoc[idx] for idx in
                         branchIdx_inRSM]  # index of the three branch out points w.r.t model.layers

    # # trainable variables do not contain MaxPooling and Flatten layers
    # # trainable variables only contain Conv and Dense layer
    # # example of WeightSizeByTrainableVar: [400, 264, 264, 288, 2112, 130], the sizes of first 3 Conv layers and then 3 Dense layers
    var = model.trainable_variables
    WeightSizeByTrainableVar = [F.count_params(i) + F.count_params(j) for i, j in zip(var[0::2], var[1::2])]

    # # because we use switching overhead reduction w.r.t time
    # # switching overhead reduction has two parts: computational savings + weight-reloading savings

    # # weight-reloading savings


def clustering(RSM, Idx, N=5):
    '''
    enumerate all possible trees and calculate the similarity score for each tree
    :param RSM: pre-computed Representation Similarity Matrix - the paper calls it task affinity tensor A
    :param N: the number of tasks
    '''

    tasks = list(range(N))
    queue = []  # to save all possible trees, and their score and model size

    sizeByBlock = get_weightsize_byBlock(Idx=Idx)

    for clusters0 in partition(tasks):

        ''' Variable Name Explanation 
        clustersX, depthX, scoreX: 'X' means the Xth branch out point
        clustersX: means the branched-out result at Xth branch out point
        scoreX: means the similarity score for clustersX
        '''

        depth0, depth1 = Idx[0], Idx[1]
        score0 = ScoreCalc(clusters0, depth0, RSM)

        # print(clusters0)
        # print('--->')
        for clusters1 in itertools.product(*[partition(cluster) for cluster in clusters0]):
            '''
            we use itertools.product to output all combinations of possible subtrees
            please refer to: https://stackoverflow.com/questions/12935194/permutations-between-two-lists-of-unequal-length
            to understand how itertools.product(*[ ]) works
            '''

            # strip the out-layer bracket
            res = []
            for l in clusters1:
                res += l
            clusters1 = res
            # print(res)

            score1 = ScoreCalc(clusters1, depth1, RSM)

            '''
            our paper only has three layers of branch out points, for the final one, it is always branched into single tasks
            so, we do not need to continue the processing of clusters2/score2
            clusters2 will all be single tasks, e.g. clusters2 should always be [[0],[1],[2],[3],[4]] if we have 5 tasks in total. 
            then score2 will always be 0; 
            '''

            # # summarize all info and pack them into the queue

            # # on April 2022, we decided to use similarity score to evaluate a tree, the higher the better
            score = score0 + score1  # total similarity score of all branch out points, score2 is omitted as it is 0
            model_size = sizeByBlock[0] + len(clusters0) * sizeByBlock[1] \
                         + len(clusters1) * sizeByBlock[2] \
                         + len(tasks) * sizeByBlock[3]  # total model size this tree requires
            tree = [clusters0, clusters1]  # decomposition details

            queue.append([score, model_size, tree])

        # print('\n\n')

    # first sort by model size x[1] in ascending order, then by similarity score x[0] in descending order
    queue.sort(key=lambda x: (x[1], -x[0]))

    return queue


def plotQueue(queue, Type=1):
    '''
    Plot how similarity scores vary among all possible budgets
    :param queue: queue must be first processed by CalcSwitchOverhead if Type = 2
    :param Type: 1 - plot dissimilarity score; 2 - plot switch overhead
    :return:
    '''
    dic = defaultdict(list)
    for q in queue:  # q = [similarity score, model size, decomposition detail, switch overhead]
        if Type == 1:
            dic[q[1]].append(q[0])
        else:
            dic[q[1]].append(q[3])

    for idx, key in enumerate(dic.keys()):
        plt.boxplot(dic[key], positions=[idx], showfliers=False)  # do not plot outliers

    plt.title('SavingCpt  -  Idx=[1,2,4]')
    plt.ylabel('Overhead reduction in time (s)')
    plt.xlabel('Budget')
    plt.show()


def optimalTree(queue):
    '''
    on June 20, 2022, we start to variety score and cost

    find out the optimal tree (with the highest similarity score or the lowest variety score) for all budgets (all possible model sizes)
    each budget (model size) has one optimal tree which has the highest similarity score or the lowest variety score
    :param queue: queue must have already been sorted - queue.sort(key=lambda x: (x[1], x[0]))
    :return:
    '''
    # queue must be sorted first by model size (in ascending order) and then by similarity score (in descending order)
    print("Finding the optimal Tree...")

    dic = defaultdict(list)
    variety_score, budget, cost = [], [], []

    # on June 20, 2022, we use cost, so we have to sort by x[3] (cost) in ascending order
    # first sort by budget x[1] then sort by cost x[3]
    queue.sort(key=lambda x: (x[1], x[3]))

    dic = defaultdict(list)
    variety_score = []
    for q in queue:       # q = [similarity score, model size, decomposition detail, cost, overhead]
        if q[1] not in dic:
            # q[1] is model size - we use it as dic's key
            dic[q[1]].append(2 - q[0])  # q[0] - similarity score of a tree, variety score = 2 - similarity score
            dic[q[1]].append(q[3])  # q[3] - cost
            dic[q[1]].append(q[4])  # q[4] - overhead
            dic[q[1]].append(q[2])  # q[2] - decomposition detail of two middle layers

            variety_score.append(2 - q[0])   # variety score = 2 - similarity score
            budget.append(q[1])
            cost.append(q[3])

    print('\nPrinting out results:\nBudget   Variety_score   Cost    Overhead   Decomposition_detail')
    for key, value in dic.items():
        print('{:<9}{:.5f}         {:5.2f}   {:.3f}      {}'.format(key, value[0], value[1], value[2], value[3]))

    # # plot score v.s. overhead
    '''
    Attention: we may have to remove the first few scores to make the curve smoother to avoid abrupt slop 
    '''
    remove = 1
    variety_score, cost, budget = variety_score[remove:], cost[remove:], budget[remove:]

    # whether you want to smooth the line or not
    smooth = True  # True or False
    if smooth:
        variety_score, cost = descending(np.array(variety_score)), ascending(np.array(cost))  # smooth the line
    else:
        variety_score, cost = np.array(variety_score), np.array(cost) # not smooth the line

    # # normalize by max-min normalization method
    score = list((variety_score - variety_score.min()) / (variety_score.max() - variety_score.min()))
    overhead = list((cost - cost.min()) / (cost.max() - cost.min()))


    return score, overhead, budget


def CalcSwitchOverheadReduction(queue, Idx, reduction=0):
    '''
    Calculate switch overhead reduction for each tree in queue
    switch overhead reduction has two parts: computational saving in time + weights-reloading saving in time
    :param queue: a list of trees
    :param Idx: location/index of the three branch out points, w.r.t RSM
    :return: no return, directly append the calculated result to queue at each row
    '''

    '''
    [an old way of calculating switching overhead, it does not apply anymore. But the background logic is the same.]
    [help me understand how the total switching overhead reduction is calculated now]
    we can use a symmetric matrix to store the task-wise switch overhead
    the maximum overhead from one task to another is to switch 3 nodes
    so the default total switch overhead for all possible pairs is 3 * the number of total pairs
    for N tasks, the number of all pairs is 1+2+...+(N-1) = sum(range(N))
    we then iterate the middle layers, if we find two tasks in one cluster, we decrease the overhead by 1
    the reminding overhead is the final total switch overhead over all possible pairs
    '''

    # # cpt_byBlock, computational overhead of each block w.r.t time, we have 4 blocks, so it has 4 values
    # # wgt_byBlock, weight-reloading overhead of each block w.r.t time
    sizeByBlock = get_weightsize_byBlock(Idx)

    wgt_byBlock = [w * 2 / 64000 * 0.6 for w in
                   sizeByBlock]  # w is number of params, we use 16bit, so w * 2 is the total byte of memory
    # based on our hardware experiment, it take 600ms to read 64KB
    cpt_byBlock = get_ComputationalSavings(Idx)

    for idx, q in enumerate(queue):

        # an example of q: [0.8002560306341132, 3, [[[0, 1, 2, 3, 4, 5, 6]], [[0, 1, 2, 3, 4, 5, 6]]]]

        # # SavingCpt: computational savings w.r.t time
        # # SavingWgt: weights-reloading savings w.r.t time
        SavingWgt, SavingCpt = 0, 0  # reset for each q

        for idx2, layer in enumerate(
                q[2]):  # q contains an optimal decomposition tree's middle two blocks which is q[2]
            for cluster in layer:

                # # the decomposition tree only has two middle layers
                if idx2 == 0:  # for the 1st middle layer
                    SavingCpt += sum(range(len(cluster))) * cpt_byBlock[1]
                    SavingWgt += sum(range(len(cluster))) * wgt_byBlock[1]
                elif idx2 == 1:  # for the 2nd middle layer
                    SavingCpt += sum(range(len(cluster))) * cpt_byBlock[2]
                    SavingWgt += sum(range(len(cluster))) * wgt_byBlock[2]

        # queue[idx].append(SavingWgt + SavingCpt) # append the total savings

        if reduction == 0:
            queue[idx].append(SavingWgt + SavingCpt)
        elif reduction == 1:
            queue[idx].append(SavingWgt)  # for deciding the budget using cross point of reduction and similarity
        elif reduction == 2:
            queue[idx].append(SavingCpt)  # for deciding locations of the three branch out points




def CalcCost(queue, Idx, N):
    '''
    Jun 20, 2022
    For 2022 sensys submission, we finally decided to use cost/variety instead of overhead-reduction/similarity-score
    we now have some new definitions
    by 'cost', we mean total time or energy required to run all tasks,
    by 'overhead' itself, we mean the portion of cost that is due to switching. Before, we were using overhead to mean the cost

    :param queue: a list of trees
    :param Idx: location/index of the three branch out points, w.r.t RSM
    :param N: number of tasks
    :return: no return, directly append the calculated result to queue at each row
    '''


    def cal_Matrix(N, decomposition):
        '''
        [this function is copy-pasted from MTL_baseline.py]
        calculate a Matrix that shows the deepest shared block index among each task pair
        the cost of transferring from one task to another only depends on how deep the two tasks share blocks
        '''

        # decomposition = [   [[0, 1, 2, 3], [4]], [[0], [2], [1, 3], [4]]   ]
        # N = taskNum # number of tasks

        # # we use Matrix to show the deepest shared block index among each task pair
        Matrix = np.zeros((N, N), dtype=int)
        for i in range(N - 1):
            for j in range(i + 1, N):
                # # for each pair of tasks, we search them in the decomposition tree
                # # to see how deep they can go until they are branched out into different branches

                for idx, layer in enumerate(decomposition):
                    for cluster in layer:
                        if i in cluster and j in cluster:
                            # # decomposition only contains the two middle layers decomposition details
                            # # so when idx = 0, it actually means they share up to the (idx + 1) layer
                            Matrix[i][j] = Matrix[j][i] = idx + 1
        return Matrix

    # # cpt_byBlock, computational overhead of each block w.r.t time, we have 4 blocks, so it has 4 values
    # # wgt_byBlock, weight-reloading overhead of each block w.r.t time
    sizeByBlock = get_weightsize_byBlock(Idx)
    wgt_byBlock = [w * 2 / 64000 * 0.6 for w in sizeByBlock]  # w is number of params, we use 16bit, so w * 2 is the total byte of memory
    cpt_byBlock = get_ComputationalSavings(Idx)  # based on our hardware experiment, it take 600ms to read 64KB

    for idx, q in enumerate(queue):

        # an example of q: [0.8002560306341132, 3, [[[0, 1, 2, 3, 4, 5, 6]], [[0, 1, 2, 3, 4, 5, 6]]]]
        # q[0] - similarity score
        # q[1] - budget size
        # q[2] - decomposition detail

        # q[3] - we will append a q[3] below for saving cost
        # q[4] - we will also append a q[4] below for saving overhead

        # # overheadCpt: computational overhead w.r.t time
        # # overheadWgt: weights-reloading overhead w.r.t time
        overheadWgt, overheadCpt = 0, 0  # reset for each q

        mat = cal_Matrix(N=N, decomposition=q[2]) # calculate a matrix about pair-wise sharing depth

        cost_history = []
        overhead_history = []
        order = list(range(N))
        for iter in range(10):  # since we are using the cost which actually depends on the execution order
                                # we randomly generate some orders and use the averaged cost as the cost
                                # the more iterations you use, the stabler the final decomposition result will be but it will also be slower
                                # I tested on N = 5 / 7, using 10 iter is stable enough
            random.shuffle(order)
            cost = 0
            overhead = 0
            transition = []

            order = list(order)
            order_ext = order + [order[0]]  #  we need to append the first task to the last one to form a loop

            for t_curr, t_next in zip(order_ext, order_ext[1:]):
                SharedDepth = mat[t_curr][t_next]
                transition.append(SharedDepth)

                # cost means total time or energy required to run all tasks - inference + weight-reloading
                cost += sum(cpt_byBlock[SharedDepth+1:])
                cost += sum(wgt_byBlock[SharedDepth+1:])

                # overhead means the portion of cost that is due to switching - weight-reloading
                overhead += sum(wgt_byBlock[SharedDepth+1:])

            cost_history.append(cost)
            overhead_history.append(overhead)

        queue[idx].append(np.average(cost_history))  # append as q[3]
        queue[idx].append(np.average(overhead_history))  # append as q[4]


def plotTraderOff_oneTree(Idx, N=5):
    # plot the tradeoff between overhead reduction and similarity score to decide budget
    RSM = np.load('rsm.npy')
    print('Idx = {}'.format(Idx))

    queue = clustering(RSM, Idx, N=N)  # group tasks according to similarity score
    # CalcSwitchOverheadReduction(queue, Idx, reduction=1)  # calculate switching overhead reduction for each tree in q
    CalcCost(queue, Idx, N)
    variety_score, cost, budget = optimalTree(queue)


    fontsize = 13
    linewidth = 2

    fig, ax = plt.subplots()

    x = np.linspace(0, 1, len(variety_score))
    ax.plot(x, variety_score, 'r', label='Variety score', linewidth=linewidth)
    ax.plot(x, cost, 'b', label='Cost', linewidth=linewidth)

    ##############################################
    # ### write trade off data into excel file
    workbook = xlsxwriter.Workbook('tradeoff.xlsx')
    worksheet = workbook.add_worksheet()
    # Start from the first cell. Rows and columns are zero indexed.

    row, col = 1, 0
    # Iterate over the data and write it out row by row.
    for s, o, b in zip(variety_score, cost, budget):
        worksheet.write(row, col, s)
        worksheet.write(row, col + 1, o)
        worksheet.write(row, col + 2, b)
        row += 1
    worksheet.write(0, 0, "Score")
    worksheet.write(0, 1, "Overhead")
    worksheet.write(0, 2, "Budget")
    workbook.close()
    ################################################

    print('\n\n*********************************')
    print('X-axis - Budget')
    for i in range(len(x)):
        print('{:.3f} - {}'.format(x[i], budget[i]))

    # # plot the intersection point and a vertical line segment
    point = (0.197, 0.498)  # the coordinates of the intersection point
    # circle = plt.Circle(point, 0.02, color='green')
    # ax.add_patch(circle)
    ax.plot([point[0], point[0]], [0, point[1]], 'k', linewidth=linewidth, linestyle='dotted')  # plot a vertical line

    plt.ylim([0, 1])
    plt.xlim([0, 1])
    ax.legend(loc='right', fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('Task similarity score\nand Overhead reduction', fontsize=fontsize)
    plt.xlabel('Model Size Budget', fontsize=fontsize)
    # plt.title('Normalized Results', fontsize=fontsize)

    # bbox_to_anchor = (x0, y0, width, height)
    plt.legend(bbox_to_anchor=(-0.25, 1.0, 1.26, 0.9), loc=3, shadow=False, mode='expand', ncol=2, fontsize='large')

    fig.set_size_inches(4.5, 3.5)
    plt.subplots_adjust(
        left=0.193,
        bottom=0.14,
        right=0.951,
        top=0.877,
        wspace=1,
        hspace=0.5,
    )

    fig.show()
    fig.savefig("algo1_tradeoff.pdf")


def findLocactionOfBP(N=7, LayerNum=5, BranchNum=3):
    '''
    find the locations of the 'BranchNum' branch out points out of the 'LayerNum' possible branch points
    e.g., in NWS, most networks have 6 layers, their LayerNum should be 5 as there are 5 places where we can branch
                  we use three branches so the BranchNum is 3

    :return: e.g. Idx = [0,1,4]
    '''

    RSM = np.load('rsm.npy')

    dic_final = {}
    for Idx in itertools.combinations([i for i in range(LayerNum)], BranchNum):
        queue = clustering(RSM, Idx, N=N)  # group tasks according to similarity score

        # CalcSwitchOverheadReduction(queue, Idx, reduction=0)  # calculate overhead reduction, use SavingCpt
        CalcCost(queue, Idx, N)

        # # first sort by model size x[1] in ascending order, then by overhead reduction x[3] in descending order
        # queue.sort(key=lambda x: (x[1], -x[3]))

        queue.sort(key=lambda x: (x[1], x[3]))


        dic = defaultdict(int)
        w1, w2 = 50, 1  # w1: weight of variety score; w2: weight of cost
        for q in queue:
            if q[1] not in dic:  # q[1] is model size, q[3] is cost

                '''
                we have three ways to evaluate the BPlocaiton
                case (1) only use cost
                case (2) only use variety score
                case (3) use weighted cost and variety score
                '''

                dic[q[1]] = q[3]  # case (1)
                dic[q[1]] = 2 - q[0]  # case (2)
                dic[q[1]] = (2 - q[0]) * w1 + q[3] * w2  # case (3)


        sum_reduction = sum(dic.values())
        print(Idx, sum_reduction)



        # for v in dic.values():
        #     print(v)

        # if sum_reduction > Max_value:
        #     Max_value = sum_reduction
        #     Max_loc = Idx

        # # add current location arrangement into dic_final
        dic_final[''.join(str(i) for i in Idx)] = sum_reduction

    print('\n\n**********')
    print('N = {}'.format(N))
    print('value - location arrangement')
    for k, v in sorted(dic_final.items(), key=lambda x: x[1], reverse=False):
        print('{:.1f} - {}'.format(v, [int(i) for i in k]))



    # print('The loc with max value is: {}'.format(Max_loc))
    # return Max_loc


########################### program execution entry ############################


# for debug, we pre-train all single models in advance
# so that we do not need train them every time
for i in range(10):
    model_train(train=True, chosenType=i)


# for debug, we calculate RSM once and save it
# so that we do not need to recompute every time
rsm = RSM_Calc(50)
np.save('rsm.npy', rsm)
print('RSM saved...')


RSM = np.load('rsm.npy')
print('RSM reloaded...')
plotTraderOff_oneTree(Idx=[0,1,4], N=5)


# # find the location arrangement of branch out points
# # change the layer-wise inference time in get_ComputationalSavings() for each dataset accordingly
# print('\n\n')
# start_time = time.time()
# findLocactionOfBP(N=6, LayerNum=5, BranchNum=3)  # # for 6-layer design: LayerNum=5, BranchNum=3;
#                                                  # # for 5-layer design: LayerNum=4, BranchNum=3;
# print("--- {0:.2f} minutes ---".format((time.time() - start_time) / 60))


########################### below is debug history, you can ignore ############################

# RSM = np.load('rsm.npy')
# # start = time.time()
# # queue = clustering_withBudget(RSM, N=7, Budget=6)xky
# # end = time.time()
# # print('Time spent: {} second'.format(end - start))
# queue = clustering(RSM, N=5)
# CalcSwitchOverhead(queue)
# # plotQueue(queue, Type=2)
# optimalTree(queue)

# Idx = (1,2,4)
# N = 7
# RSM = np.load('rsm.npy')
# print('Idx = {}'.format(Idx))
# queue = clustering(RSM, Idx, N=N)  # group tasks according to similarity score
# CalcSwitchOverheadReduction(queue, Idx)
# plotQueue(queue, Type=2)
#
# # # first sort by model size x[1] in ascending order, then by overhead reduction x[3] in descending order
# queue.sort(key=lambda x: (x[1], -x[3]))
# dic = defaultdict(int)
# for q in queue:
#     if q[1] not in dic:
#         dic[q[1]] = q[3]  # log the max reduction of each budget


def fun(Idx):
    RSM = np.load('rsm.npy')

    queue = clustering(RSM, Idx, N=N)  # group tasks according to similarity score
    CalcSwitchOverheadReduction(queue, Idx, reduction=2)  # calculate overhead reduction, use SavingCpt

    # # first sort by model size x[1] in ascending order, then by overhead reduction x[3] in descending order
    queue.sort(key=lambda x: (x[1], -x[3]))

    dic = defaultdict(int)
    for q in queue:
        if q[1] not in dic:  # q[1] is model size, q[3] is overhead reduction
            dic[q[1]] = q[3]  # log the max reduction of each budget

    print(Idx, sum(dic.values()))