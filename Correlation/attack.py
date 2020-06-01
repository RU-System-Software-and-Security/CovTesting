from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent

import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import keras
from keras import backend as K
import numpy as np
import time
from util import get_model
import sys
import argparse


# def JSMA(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     jsma = SaliencyMapMethod(model_wrap, sess=sess)
#     jsma_params = {'theta':1., 'gamma': 0.1, 'clip_min':0., 'clip_max':1.}
#     adv = jsma.generate_np(x, **jsma_params)
#     return adv
#
#
# def FGSM(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     fgsm = FastGradientMethod(model_wrap, sess=sess)
#     fgsm_params={'y':y, 'eps':0.2, 'clip_min':0., 'clip_max': 1.}
#     adv = fgsm.generate_np(x, **fgsm_params)
#     return adv
#
#
# def BIM(model, x, y):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     bim = BasicIterativeMethod(model_wrap, sess=sess)
#     bim_params={'eps_iter': 0.03, 'nb_iter': 10, 'y':y, 'clip_min': 0., 'clip_max': 1.}
#     adv = bim.generate_np(x, **bim_params)
#     return adv

# # invoke the method for many times leads to multiple symbolic graph and may cause OOM
# def CW(model, x, y, batch):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     cw = CarliniWagnerL2(model_wrap, sess=sess)
#     cw_params = {'binary_search_steps': 1,
#                  'y': y,
#                  'learning_rate': .1,
#                  'max_iterations': 50,
#                  'initial_const': 10,
#                  'batch_size': batch,
#                  'clip_min': -0.5,
#                  'clip_max': 0.5}
#     adv = cw.generate_np(x, **cw_params)# invoke the method for many times leads to multiple symbolic graph and may cause OOM
# # def CW(model, x, y, batch):
# #     sess = K.get_session()
# #     model_wrap = KerasModelWrapper(model)
# #     cw = CarliniWagnerL2(model_wrap, sess=sess)
# #     cw_params = {'binary_search_steps': 1,
# #                  'y': y,
# #                  'learning_rate': .1,
# #                  'max_iterations': 50,
# #                  'initial_const': 10,
# #                  'batch_size': batch,
# #                  'clip_min': -0.5,
# #                  'clip_max': 0.5}
# #     adv = cw.generate_np(x, **cw_params)
# #     return adv
# #
# #
# # # for mnist, eps=.3, eps_iter=.03, nb_iter=10
# # # for cifar and svhn, eps=8/255, eps_iter=.01, nb_iter=30
# # def PGD(model, x, y, batch):
# #     sess = K.get_session()
# #     model_wrap = KerasModelWrapper(model)
# #     pgd = ProjectedGradientDescent(model_wrap, sess=sess)
# #     pgd_params = {'eps': 8. / 255.,
# #                   'eps_iter': .01,
# #                   'nb_iter': 30.,
# #                   'clip_min': -0.5,
# #                   'clip_max': 0.5,
# #                   'y': y}
# #     adv = pgd.generate_np(x, **pgd_params)
# #     return adv
#     return adv
#
#
# # for mnist, eps=.3, eps_iter=.03, nb_iter=10
# # for cifar and svhn, eps=8/255, eps_iter=.01, nb_iter=30
# def PGD(model, x, y, batch):
#     sess = K.get_session()
#     model_wrap = KerasModelWrapper(model)
#     pgd = ProjectedGradientDescent(model_wrap, sess=sess)
#     pgd_params = {'eps': 8. / 255.,
#                   'eps_iter': .01,
#                   'nb_iter': 30.,
#                   'clip_min': -0.5,
#                   'clip_max': 0.5,
#                   'y': y}
#     adv = pgd.generate_np(x, **pgd_params)
#     return adv


# integrate all attack method in one function and only construct graph once
def gen_adv_data(model, x, y, method, dataset, batch=2048):
    sess = K.get_session()
    model_wrap = KerasModelWrapper(model)
    if method.upper() == 'CW':
        params = {'binary_search_steps': 1,
                  'y': y,
                  'learning_rate': .1,
                  'max_iterations': 50,
                  'initial_const': 10,
                  'batch_size': batch,
                  # 'clip_min': -0.5,
                  # 'clip_max': 0.5
                  }
        attack = CarliniWagnerL2(model_wrap, sess=sess)

        data_num = x.shape[0]
        begin, end = 0, batch
        adv_x_all = np.zeros_like(x)
        # every time process batch_size
        while end < data_num:
            start_time = time.time()
            params['y'] = y[begin:end]
            adv_x = attack.generate_np(x[begin:end], **params)
            adv_x_all[begin: end] = adv_x
            print(begin, end, "done")
            begin += batch
            end += batch
            end_time = time.time()
            print("time: ", end_time - start_time)

        # process the remaining
        if begin < data_num:
            start_time = time.time()
            params['y'] = y[begin:]
            params['batch_size'] = data_num - begin
            adv_x = attack.generate_np(x[begin:], **params)
            adv_x_all[begin:] = adv_x
            print(begin, data_num, "done")
            end_time = time.time()
            print("time: ", end_time - start_time)

    elif method.upper() == 'PGD':
        if dataset == 'cifar':
            params = {'eps': 16. / 255.,
                      'eps_iter': 2. / 255.,
                      'nb_iter': 30.,
                      # 'clip_min': -0.5,
                      # 'clip_max': 0.5,
                      'y': y}
            attack = ProjectedGradientDescent(model_wrap, sess=sess)
        elif dataset == 'mnist':
            params = {'eps': .3,
                      'eps_iter': .03,
                      'nb_iter': 20.,
                      'clip_min': -0.5,
                      'clip_max': 0.5,
                      'y': y}
            attack = ProjectedGradientDescent(model_wrap, sess=sess)
        elif dataset == 'svhn':
            params = {'eps': 8. / 255.,
                      'eps_iter': 0.01,
                      'nb_iter': 30.,
                      'clip_min': -0.5,
                      'clip_max': 0.5,
                      'y': y}
            attack = ProjectedGradientDescent(model_wrap, sess=sess)

        data_num = x.shape[0]
        begin, end = 0, batch
        adv_x_all = np.zeros_like(x)
        # every time process batch_size
        while end < data_num:
            start_time = time.time()
            params['y'] = y[begin:end]
            adv_x = attack.generate_np(x[begin:end], **params)
            adv_x_all[begin: end] = adv_x
            print(begin, end, "done")
            begin += batch
            end += batch
            end_time = time.time()
            print("time: ", end_time - start_time)

        # process the remaining
        if begin < data_num:
            start_time = time.time()
            params['y'] = y[begin:]
            adv_x = attack.generate_np(x[begin:], **params)
            adv_x_all[begin:] = adv_x
            print(begin, data_num, "done")
            end_time = time.time()
            print("time: ", end_time - start_time)

    else:
        print('Unsupported attack')
        sys.exit(1)

    return adv_x_all


# def gen_adv_data(model, x, y, method, batch=4000):
#     data_num = x.shape[0]
#     begin, end = 0, batch
#     adv_x_all = np.zeros_like(x)
#     while begin < data_num:
#         start_time = time.time()
#         adv_x = method(model, x[begin:end], y[begin:end], end-begin)
#         adv_x_all[begin: end] = adv_x
#         print(begin, end, "done")
#         begin += batch
#         end += batch
#         if end >= data_num:
#             end = data_num
#         end_time = time.time()
#         print("time: ", end_time - start_time)
#     return adv_x_all


# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('./data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('./data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('./data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('./data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def accuracy(model, x, labels):
    assert (x.shape[0] == labels.shape[0])
    num = x.shape[0]
    y = model.predict(x)
    y = y.argmax(axis=-1)
    labels = labels.argmax(axis=-1)
    idx = (labels == y)
    print(np.sum(idx) / num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attack for DNN')
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-model', help="target model to attack", choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'adv_lenet1', 'adv_lenet4', 'adv_lenet5', 'adv_vgg16', 'adv_resnet20', 'svhn_model', 'adv_svhn_model', 'svhn_first', 'adv_svhn_first', 'svhn_second','adv_svhn_second'])
    parser.add_argument('-attack', help="attack model", choices=['CW', 'PGD'])
    parser.add_argument('-batch_size', help="attack batch size", type=int, default=32)

    args = parser.parse_args()
    # args.dataset = 'cifar'
    # args.attack = 'PGD'

    # ## get MNIST or SVHN
    x_train, y_train, x_test, y_test = load_data(args.dataset)

    ## get CIFAR
    # from util import get_data
    # x_train, y_train, x_test, y_test = get_data(args.dataset)

    # ## load Xuwei's trained svhn model
    # model = get_model(args.dataset, True)
    # model.load_weights('./data/' + args.dataset + '_data/model/' + args.model + '.h5')
    # model.compile(
    #     loss='categorical_crossentropy',
    #     # optimizer='adadelta',
    #     optimizer='adam',
    #     metrics=['accuracy']
    # )

    ## load mine trained model
    from keras.models import load_model
    model = load_model('./data/' + args.dataset + '_data/model/' + args.model + '.h5')
    model.summary()

    accuracy(model, x_test, y_test)

    adv = gen_adv_data(model, x_test, y_test, args.attack, args.dataset, args.batch_size)

    # accuracy(model, adv, y_test)
    # np.save('./data/cifar_data/model/test_adv_PGD', adv)
    np.save('./data/' + args.dataset + '_data/model/' + args.model + '_' + args.attack + '.npy', adv)

    # y_res = model.predict(x_train)
    # y_res = softmax(y_res)
    # y_res = y_res.argmax(axis=-1)
    # y = y_train.argmax(axis=-1)
    # idx = (y_res == y)
