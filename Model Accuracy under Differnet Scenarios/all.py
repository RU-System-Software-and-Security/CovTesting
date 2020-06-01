import argparse
import os
import random
import shutil
import warnings
import sys

warnings.filterwarnings("ignore")

from keras import backend as K
import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM
import keras
from util import get_model

import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('./data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('./data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('./data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('./data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test

class AttackEvaluate:
    # model does not have softmax layer
    def __init__(self, model, ori_x, ori_y, adv_x):
        self.model = model
        # get the raw data
        self.nature_samples = ori_x
        self.labels_samples = ori_y
        # get the adversarial examples
        self.adv_samples = adv_x
        # self.adv_labels = np.load('{}{}_AdvLabels.npy'.format(self.AdvExamplesDir, self.AttackName))

        predictions = model.predict(self.adv_samples)

        def soft_max(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        tmp_soft_max = []
        for i in range(len(predictions)):
            tmp_soft_max.append(soft_max(predictions[i]))

        self.softmax_prediction = np.array(tmp_soft_max)

    # help function
    def successful(self, adv_softmax_preds, nature_true_preds):
        if np.argmax(adv_softmax_preds) != np.argmax(nature_true_preds):
            return True
        else:
            return False

    # 1 MR:Misclassification Rate
    def misclassification_rate(self):

        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
        mr = cnt / len(self.adv_samples)
        print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MR and Linf')
    parser.add_argument('-dataset', help="dataset to use", choices=['mnist', 'cifar', 'svhn'])
    parser.add_argument('-model', help="target model to attack",
                        choices=['vgg16', 'resnet20', 'lenet1', 'lenet4', 'lenet5', 'svhn_model',
                                 'svhn_first', 'svhn_second'])
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    attack = 'PGD'

    ### benign dataset
    x_train, y_train, x_test, y_test = load_data(dataset)

    ### deephunter dataset
    x_adv_dp = np.load('./data/' + dataset + '_data/model/' + 'deephunter_adv_test_{}.npy'.format(model_name))

    ### pgd dataset
    x_adv_pgd = np.load('./data/' + dataset + '_data/model/' + model_name + '_' + attack + '.npy')



    # ## load benign model
    from keras.models import load_model

    model_benign = load_model('./data/' + dataset + '_data/model/' + model_name + '.h5')
    model_benign.summary()

    # ## load deephunter model
    from keras.models import load_model

    model_dp = load_model('new_model/dp_{}.h5'.format(model_name))
    model_dp.summary()

    ##### load pgd trained model
    model_pgd = load_model('./data/' + dataset + '_data/model/adv_' + model_name + '.h5')
    model_pgd.summary()


    ##############################Q1########################
    criteria1_1 = AttackEvaluate(model_benign, x_test, y_test, x_test)
    MR1_1 = 1 - criteria1_1.misclassification_rate()

    criteria1_2 = AttackEvaluate(model_dp, x_test, y_test, x_test)
    MR1_2 = 1 - criteria1_2.misclassification_rate()


    ##############################Q2#########################
    criteria2_1 = AttackEvaluate(model_benign, x_test, y_test, x_adv_dp)
    MR2_1 = 1 - criteria2_1.misclassification_rate()

    criteria2_2 = AttackEvaluate(model_dp, x_test, y_test, x_adv_dp)
    MR2_2 = 1 - criteria2_2.misclassification_rate()


    #############################Q3##########################
    criteria3 = AttackEvaluate(model_dp, x_test, y_test, x_adv_pgd)
    MR3 = 1 - criteria3.misclassification_rate()


    #############################Q4##########################
    criteria4 = AttackEvaluate(model_pgd, x_test, y_test, x_adv_dp)
    MR4 = 1 - criteria4.misclassification_rate()

    print(MR1_1, MR1_2)
    print(MR2_1, MR2_2)
    print(MR3)
    print(MR4)

    with open("result.txt", "a") as f:
        f.write("\n------------------------------------------------------------------------------\n")
        f.write('the result of {} {} is: \n'.format(args.dataset, args.model))
        f.write('Benign-dh is {} and ({})\n'.format(MR1_2, MR1_2-MR1_1))
        f.write('DH-dh is {} and ({})\n'.format(MR2_2, MR2_2-MR2_1))
        f.write('PGD-dh is {} \n'.format(MR3))
        f.write('DH-pgd is {} and ({})\n'.format(MR4, MR4-MR2_1))







