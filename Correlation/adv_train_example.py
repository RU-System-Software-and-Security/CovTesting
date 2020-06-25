from util import get_data, get_model

from keras.preprocessing.image import ImageDataGenerator
from art.data_generators import KerasDataGenerator

from art.classifiers import KerasClassifier
from art.attacks import ProjectedGradientDescent
from art.attacks import BasicIterativeMethod
from art.defences import AdversarialTrainer
from art.attacks import CarliniL2Method
from art.attacks import FastGradientMethod

import numpy as np
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
    x_train = np.load('../data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('../data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('../data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('../data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    dataset = 'svhn'
    model_name = 'svhn_second'
    attack_name = 'PGD'

    # dataset = 'cifar'
    # model_name = 'resnet20'
    # attack_name = 'PGD'

    # x_train, y_train, x_test, y_test = get_data('cifar')
    x_train, y_train, x_test, y_test = load_data(dataset)


    from keras.models import load_model
    model = load_model('../data/' + dataset + '_data/model/' + model_name + '.h5')

    # ## for svhn model
    # from util import get_model
    # model = get_model(dataset, True)
    # model.load_weights('./data/' + dataset + '_data/model/' + model_name + '.h5')

    model.compile(
        loss='categorical_crossentropy',
        # optimizer='adadelta',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    # Evaluate the benign trained model on clean test set
    labels_true = np.argmax(y_test, axis=1)
    labels_test = np.argmax(model.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

    # # training for MNIST
    # classifier = KerasClassifier(clip_values=(-0.5, 0.5), model=model, use_logits=False)
    # attack = ProjectedGradientDescent(classifier, eps=0.3, eps_step=0.01, max_iter=20, batch_size=128)

    # ## training for CIFAR
    # classifier = KerasClassifier(model=model, use_logits=False)
    # attack = ProjectedGradientDescent(classifier, eps=8/255, eps_step=2/255, max_iter=10, batch_size=512)

    ## training for SVHN
    classifier = KerasClassifier(clip_values=(-0.5, 0.5), model=model, use_logits=False)
    attack = ProjectedGradientDescent(classifier, eps=8/255, eps_step=1/255, max_iter=20, batch_size=512)

    x_test_pgd = attack.generate(x_test, y_test)
    # np.save('./data/' + dataset + '_data/model/' + model_name + '_y_' + attack_name + '.npy', x_test_pgd)

    # Evaluate the benign trained model on adv test set
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on original PGD adversarial samples: %.2f%%' %
          (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))

    trainer = AdversarialTrainer(classifier, attack, ratio=1.0)
    trainer.fit(x_train, y_train, nb_epochs=60, batch_size=1024)

    classifier.save(filename='adv_' + model_name + '.h5', path='../data/' + dataset + '_data/model/')

    # Evaluate the adversarially trained model on clean test set
    labels_true = np.argmax(y_test, axis=1)
    labels_test = np.argmax(classifier.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

    # Evaluate the adversarially trained model on original adversarial samples
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on original PGD adversarial samples: %.2f%%' %
          (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))

    # Evaluate the adversarially trained model on fresh adversarial samples produced on the adversarially trained model
    x_test_pgd = attack.generate(x_test, y_test)
    labels_pgd = np.argmax(classifier.predict(x_test_pgd), axis=1)
    print('Accuracy on new PGD adversarial samples: %.2f%%' % (np.sum(labels_pgd == labels_true) / x_test.shape[0] * 100))





