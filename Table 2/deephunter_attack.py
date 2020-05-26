from mutators import Mutators
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from keras import backend as K

from keras import backend as K

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import argparse
import os
import random
import shutil
import warnings
import sys

warnings.filterwarnings("ignore")

# helper function
def get_layer_i_output(model, i, data):
    layer_model = K.function([model.layers[0].input], [model.layers[i].output])
    ret = layer_model([data])[0]
    num = data.shape[0]
    ret = np.reshape(ret, (num, -1))
    return ret


class Coverage:
    def __init__(self, model, x_train, y_train, x_test, y_test, x_adv):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_adv = x_adv

    # find scale factors and min num
    def scale(self, layers, batch=1024):
        data_num = self.x_adv.shape[0]
        factors = dict()
        for i in layers:
            begin, end = 0, batch
            max_num, min_num = np.NINF, np.inf
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                tmp = layer_output.max()
                max_num = tmp if tmp > max_num else max_num
                tmp = layer_output.min()
                min_num = tmp if tmp < min_num else min_num
                begin += batch
                end += batch
            factors[i] = (max_num - min_num, min_num)
        return factors

    # 1 Neuron Coverage
    def NC(self, layers, threshold=0., batch=1024):
        factors = self.scale(layers, batch=batch)
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        activate_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            buckets = np.zeros(neurons).astype('bool')
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                # scale the layer output to (0, 1)
                layer_output -= factors[i][1]
                layer_output /= factors[i][0]
                col_max = np.max(layer_output, axis=0)
                begin += batch
                end += batch
                buckets[col_max > threshold] = True
            activate_num += np.sum(buckets)
        # print('NC:\t{:.3f} activate_num:\t{} neuron_num:\t{}'.format(activate_num / neuron_num, activate_num, neuron_num))
        return activate_num / neuron_num, activate_num, neuron_num

    # 2 k-multisection neuron coverage, neuron boundary coverage and strong activation neuron coverage
    def KMNC(self, layers, k=10, batch=1024):
        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        covered_num = 0
        l_covered_num = 0
        u_covered_num = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            print(neurons)
            begin, end = 0, batch
            data_num = self.x_train.shape[0]

            neuron_max = np.full(neurons, np.NINF).astype('float')
            neuron_min = np.full(neurons, np.inf).astype('float')
            while begin < data_num:
                layer_output_train = get_layer_i_output(self.model, i, self.x_train[begin:end])
                batch_neuron_max = np.max(layer_output_train, axis=0)
                batch_neuron_min = np.min(layer_output_train, axis=0)
                neuron_max = np.maximum(batch_neuron_max, neuron_max)
                neuron_min = np.minimum(batch_neuron_min, neuron_min)
                begin += batch
                end += batch
            buckets = np.zeros((neurons, k + 2)).astype('bool')
            interval = (neuron_max - neuron_min) / k
            # print(interval[8], neuron_max[8], neuron_min[8])
            begin, end = 0, batch
            data_num = self.x_adv.shape[0]
            while begin < data_num:
                layer_output_adv = get_layer_i_output(model, i, self.x_adv[begin: end])
                layer_output_adv -= neuron_min
                layer_output_adv /= (interval + 10 ** (-100))
                layer_output_adv[layer_output_adv < 0.] = -1
                layer_output_adv[layer_output_adv >= k / 1.0] = k
                layer_output_adv = layer_output_adv.astype('int')
                # index 0 for lower, 1 to k for between, k + 1 for upper
                layer_output_adv = layer_output_adv + 1
                for j in range(neurons):
                    uniq = np.unique(layer_output_adv[:, j])
                    # print(layer_output_adv[:, j])
                    buckets[j, uniq] = True
                begin += batch
                end += batch
            covered_num += np.sum(buckets[:, 1:-1])
            u_covered_num += np.sum(buckets[:, -1])
            l_covered_num += np.sum(buckets[:, 0])
        print('KMNC:\t{:.3f} covered_num:\t{}'.format(covered_num / (neuron_num * k), covered_num))
        print(
            'NBC:\t{:.3f} l_covered_num:\t{}'.format((l_covered_num + u_covered_num) / (neuron_num * 2), l_covered_num))
        print('SNAC:\t{:.3f} u_covered_num:\t{}'.format(u_covered_num / neuron_num, u_covered_num))
        return covered_num / (neuron_num * k), (l_covered_num + u_covered_num) / (
                    neuron_num * 2), u_covered_num / neuron_num, covered_num, l_covered_num, u_covered_num, neuron_num * k

    # 3 top-k neuron coverage
    def TKNC(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        neuron_num = 0
        for i in layers:
            out_shape = self.model.layers[i].output.shape
            neuron_num += np.prod(out_shape[1:])
        neuron_num = int(neuron_num)

        pattern_num = 0
        data_num = self.x_adv.shape[0]
        for i in layers:
            pattern_set = set()
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                # topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                for j in range(topk.shape[0]):
                    pattern_set.add(tuple(topk[j]))
                begin += batch
                end += batch
            pattern_num += len(pattern_set)
        print(
            'TKNC:\t{:.3f} pattern_num:\t{} neuron_num:\t{}'.format(pattern_num / neuron_num, pattern_num, neuron_num))
        return pattern_num / neuron_num, pattern_num, neuron_num

    # 4 top-k neuron patterns
    def TKNP(self, layers, k=2, batch=1024):
        def top_k(x, k):
            ind = np.argpartition(x, -k)[-k:]
            return ind[np.argsort((-x)[ind])]

        def to_tuple(x):
            l = list()
            for row in x:
                l.append(tuple(row))
            return tuple(l)

        pattern_set = set()
        layer_num = len(layers)
        data_num = self.x_adv.shape[0]
        patterns = np.zeros((data_num, layer_num, k))
        layer_cnt = 0
        for i in layers:
            neurons = np.prod(self.model.layers[i].output.shape[1:])
            begin, end = 0, batch
            while begin < data_num:
                layer_output = get_layer_i_output(self.model, i, self.x_adv[begin:end])
                topk = np.argpartition(layer_output, -k, axis=1)[:, -k:]
                topk = np.sort(topk, axis=1)
                # or in order
                # topk = np.apply_along_axis[lambda x: top_k(layer_output, k), 1, layer_output]
                patterns[begin:end, layer_cnt, :] = topk
                begin += batch
                end += batch
            layer_cnt += 1

        for i in range(patterns.shape[0]):
            pattern_set.add(to_tuple(patterns[i]))
        pattern_num = len(pattern_set)
        print('TKNP:\t{:.3f}'.format(pattern_num))
        return pattern_num

    def all(self, layers, batch=100):
        self.NC(layers, batch=batch)
        self.KMNC(layers, batch=batch)
        self.TKNC(layers, batch=batch)
        self.TKNP(layers, batch=batch)


def mutate(img):
    # ref_img is the reference image, img is the seed

    # cl means the current state of transformation
    # 0 means it can select both of Affine and Pixel transformations
    # 1 means it only select pixel transformation because an Affine transformation has been used before

    # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
    # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

    # tyr_num is the maximum number of trials in Algorithm 2

    transformations = [Mutators.image_translation, Mutators.image_scale, Mutators.image_shear, Mutators.image_rotation,
                       Mutators.image_contrast, Mutators.image_brightness, Mutators.image_blur,
                       Mutators.image_pixel_change,
                       Mutators.image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(range(-3, 3)))  # image_translation
    params.append(list(map(lambda x: x * 0.1, list(range(7, 12)))))  # image_scale
    params.append(list(map(lambda x: x * 0.1, list(range(-6, 6)))))  # image_shear
    params.append(list(range(-50, 50)))  # image_rotation
    params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
    params.append(list(range(-20, 20)))  # image_brightness
    params.append(list(range(1, 10)))  # image_blur
    params.append(list(range(1, 10)))  # image_pixel_change
    params.append(list(range(1, 4)))  # image_noise

    classA = [7, 8]  # pixel value transformation
    classB = [0, 1, 2, 3, 4, 5, 6]  # Affine transformation


    x, y, z = img.shape
    random.seed(time.time())

    tid = random.sample(classA + classB, 1)[0]
    # tid = 8
    # Randomly select one transformation   Line-7 in Algorithm2
    transformation = transformations[tid]
    params = params[tid]
    # Randomly select one parameter Line 10 in Algo2
    param = random.sample(params, 1)[0]

    # Perform the transformation  Line 11 in Algo2

    # plt.imshow(img + 0.5)
    # plt.show()

    image = np.uint8(np.round((img + 0.5) * 255))
    img_new = transformation(copy.deepcopy(image), param)/ 255.0 - 0.5
    # img_new = np.round(img_new)
    img_new = img_new.reshape(img.shape)

    # # for cifar dataset
    # img_new = transformation(img, param)
    # # img_new = np.round(img_new)
    # img_new = img_new.reshape(img.shape)

    # Otherwise the mutation is failed. Line 20 in Algo 2
    return img_new

# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('./data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('./data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('./data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('./data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test



if __name__ == '__main__':
    dataset = 'mnist'
    model_name = 'lenet1'
    # model_layer = 18

    # load dataset
    x_train, y_train, x_test, y_test = load_data(dataset)

    # import model
    from keras.models import load_model
    model = load_model('./data/' + dataset + '_data/model/' + model_name + '.h5')
    model.summary()

    x_adv = np.array([])
    for i in range(1000):
        new_image = mutate(x_test[i])

        if x_adv.size == 0:
            x_adv = np.expand_dims(new_image, axis=0)
        else:
            x_adv = np.concatenate((x_adv, np.expand_dims(new_image, axis=0)), axis=0)

    print(x_adv.shape)
    np.save('./data/' + dataset + '_data/model/' + 'deephunter_adv_test_{}.npy'.format(model_name), x_adv)













