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


# (row, col, channel)
def gaussian_blur_transform(AdvSample, radius):
    if AdvSample.shape[2] == 3:
        sample = np.round((AdvSample + 0.5) * 255)

        image = Image.fromarray(np.uint8(sample))
        gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        gb_image = np.array(gb_image).astype('float32') / 255.0 - 0.5
        # print(gb_image.shape)

        return gb_image
    else:
        sample = np.round((AdvSample + 0.5) * 255)
        sample = np.squeeze(sample, axis=2)
        image = Image.fromarray(np.uint8(sample))
        gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        gb_image = np.expand_dims(np.array(gb_image).astype('float32'), axis=-1) / 255.0 - 0.5
        # print(gb_image.shape)
        return gb_image


# use PIL Image save instead of guetzli
def image_compress_transform(IndexAdv, AdvSample, dir_name, quality=50):
    if AdvSample.shape[2] == 3:
        sample = np.round((AdvSample + .5) * 255)
        image = Image.fromarray(np.uint8(sample))
        saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
        image.save(saved_adv_image_path, format='JPEG', quality=quality)
        IC_image = Image.open(saved_adv_image_path).convert('RGB')
        IC_image = np.array(IC_image).astype('float32') / 255.0 - .5
        return IC_image
    else:
        sample = np.round((AdvSample + .5) * 255)
        sample = np.squeeze(sample, axis=2)
        image = Image.fromarray(np.uint8(sample), mode='L')
        saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
        image.save(saved_adv_image_path, format='JPEG', quality=quality)
        IC_image = Image.open(saved_adv_image_path).convert('L')
        IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=-1) / 255.0 - .5
        return IC_image
#
# # (row, col, channel)
# def gaussian_blur_transform(AdvSample, radius):
#     if AdvSample.shape[2] == 3:
#         sample = np.round((AdvSample + 0.5) * 255)
#
#         image = Image.fromarray(np.uint8(sample))
#         gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         gb_image = np.array(gb_image).astype('float32') / 255.0 - 0.5
#         # print(gb_image.shape)
#
#         return gb_image
#     else:
#         sample = np.round((AdvSample + 0.5) * 255)
#         sample = np.squeeze(sample, axis=2)
#         image = Image.fromarray(np.uint8(sample))
#         gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         gb_image = np.expand_dims(np.array(gb_image).astype('float32'), axis=-1) / 255.0 - 0.5
#         # print(gb_image.shape)
#         return gb_image

# # (row, col, channel)
# def gaussian_blur_transform(AdvSample, radius):
#     if AdvSample.shape[2] == 3:
#         sample = np.round(AdvSample)
#
#         image = Image.fromarray(np.uint8(sample))
#         gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         gb_image = np.array(gb_image).astype('float32')
#         # print(gb_image.shape)
#
#         return gb_image
#     else:
#         sample = np.round(AdvSample)
#         sample = np.squeeze(sample, axis=2)
#         image = Image.fromarray(np.uint8(sample))
#         gb_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
#         gb_image = np.expand_dims(np.array(gb_image).astype('float32'), axis=-1)
#         # print(gb_image.shape)
#         return gb_image
#
#
#
# # use PIL Image save instead of guetzli
# def image_compress_transform(IndexAdv, AdvSample, dir_name, quality=50):
#     if AdvSample.shape[2] == 3:
#         sample = np.round(AdvSample)
#         image = Image.fromarray(np.uint8(sample))
#         saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
#         image.save(saved_adv_image_path, format='JPEG', quality=quality)
#         IC_image = Image.open(saved_adv_image_path).convert('RGB')
#         IC_image = np.array(IC_image).astype('float32')
#         return IC_image
#     else:
#         sample = np.round(AdvSample)
#         sample = np.squeeze(sample, axis=2)
#         image = Image.fromarray(np.uint8(sample), mode='L')
#         saved_adv_image_path = os.path.join(dir_name, '{}th-adv.jpg'.format(IndexAdv))
#         image.save(saved_adv_image_path, format='JPEG', quality=quality)
#         IC_image = Image.open(saved_adv_image_path).convert('L')
#         IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=-1)
#         return IC_image


'''
def image_compress_transform(IndexAdv, AdvSample, dir_name, quality):
    if AdvSample.shape[-1] == 3:
        image = Image.fromarray(np.uint8(np.round((AdvSample + 0.5) * 255)))
        saved_adv_image_path = os.path.join(dir_name, '{}th-adv.png'.format(IndexAdv))
        image.save(saved_adv_image_path)
        output_IC_path = os.path.join(dir_name, '{}th-IC-adv.jpg'.format(IndexAdv))

        cmd = 'guetzli --quality {} {} [}'.format(quality, saved_adv_image_path, output_IC_path)
        assert(os.system(cmd) == 0)
        IC_image = Image.open(output_IC_path).convert('RGB')
        IC_image = np.array(IC_image).astype('float32') / 255.0 -0.5
        return IC_image
    else:
        sample = np.round((AdvSample + 0.5) * 255)
        sample = np.squeeze(sample, axis=2)
        image = Image.fromarray(np.uint8(sample), mode='L')
        saved_adv_image_path = os.path.join(dir_name, '{}th-adv.png'.format(IndexAdv))

        image.save(saved_adv_image_path)
        output_IC_path = os.path.join(dir_name, '{}th-IC-adv.jpg'.format(IndexAdv))
        cmd = 'guetzli --quality {} {} {}'.format(quality, saved_adv_image_path, output_IC_path) 
        assert(os.system(cmd) == 0)
        IC_image = Image.open(output_IC_path).convert('L')
        IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=-1) / 255.0 - 0.5
        return IC_image


# help function for the image compression transformation of images
def image_compress_transform(IndexAdv, AdvSample, dir_name, quality, oriDataset):
    if oriDataset.upper() == 'CIFAR10':
        assert AdvSample.shape == (3, 32, 32)
        sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
        image = Image.fromarray(np.uint8(sample))

        saved_adv_image_path = os.path.join(dir_name, '{}th-adv-cifar.png'.format(IndexAdv))
        image.save(saved_adv_image_path)
        output_IC_path = os.path.join(dir_name, '{}th-IC-adv-cifar.jpg'.format(IndexAdv))

        cmd = 'guetzli --quality {} {} {}'.format(quality, saved_adv_image_path, output_IC_path)
        assert os.system(cmd) == 0, 'guetzli tool should be install before, https://github.com/google/guetzli'

        IC_image = Image.open(output_IC_path).convert('RGB')
        IC_image = np.transpose(np.array(IC_image), (2, 0, 1)).astype('float32') / 255.0
        return IC_image

    if oriDataset.upper() == 'MNIST':
        assert AdvSample.shape == (1, 28, 28)
        sample = np.transpose(np.round(AdvSample * 255), (1, 2, 0))
        sample = np.squeeze(sample, axis=2)  # for MNIST, there is no RGB
        image = Image.fromarray(np.uint8(sample), mode='L')

        saved_adv_image_path = os.path.join(dir_name, '{}th-adv-mnist.png'.format(IndexAdv))
        image.save(saved_adv_image_path)
        output_IC_path = os.path.join(dir_name, '{}th-IC-adv-mnist.jpg'.format(IndexAdv))

        cmd = 'guetzli --quality {} {} {}'.format(quality, saved_adv_image_path, output_IC_path)
        assert os.system(cmd) == 0, 'guetzli tool should be install before, https://github.com/google/guetzli'

        IC_image = Image.open(output_IC_path).convert('L')
        IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=0) / 255.0
        return IC_image
'''


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

    # 2 ACAC: average confidence of adversarial class
    def avg_confidence_adv_class(self):
        cnt = 0
        conf = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                conf += np.max(self.softmax_prediction[i])

        print('ACAC:\t{:.3f}'.format(conf / cnt))
        return conf / cnt

    # 3 ACTC: average confidence of true class
    def avg_confidence_true_class(self):

        true_labels = np.argmax(self.labels_samples, axis=1)
        cnt = 0
        true_conf = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                true_conf += self.softmax_prediction[i, true_labels[i]]
        print('ACTC:\t{:.3f}'.format(true_conf / cnt))
        return true_conf / cnt

    # 4 ALP: Average L_p Distortion
    def avg_lp_distortion(self):

        ori_r = np.round(self.nature_samples * 255)
        adv_r = np.round(self.adv_samples * 255)

        NUM_PIXEL = int(np.prod(self.nature_samples.shape[1:]))

        pert = adv_r - ori_r

        dist_l0 = 0
        dist_l2 = 0
        dist_li = 0

        cnt = 0

        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                dist_l0 += (np.linalg.norm(np.reshape(pert[i], -1), ord=0) / NUM_PIXEL)
                dist_l2 += np.linalg.norm(np.reshape(self.nature_samples[i] - self.adv_samples[i], -1), ord=2)
                dist_li += np.linalg.norm(np.reshape(self.nature_samples[i] - self.adv_samples[i], -1), ord=np.inf)

        adv_l0 = dist_l0 / cnt
        adv_l2 = dist_l2 / cnt
        adv_li = dist_li / cnt

        print('**ALP:**\n\tL0:\t{:.3f}\n\tL2:\t{:.3f}\n\tLi:\t{:.3f}'.format(adv_l0, adv_l2, adv_li))
        return adv_l0, adv_l2, adv_li

    # 5 ASS: Average Structural Similarity
    def avg_SSIM(self):

        ori_r_channel = np.round(self.nature_samples * 255).astype(dtype=np.float32)
        adv_r_channel = np.round(self.adv_samples * 255).astype(dtype=np.float32)

        totalSSIM = 0
        cnt = 0

        """
        For SSIM function in skimage: http://scikit-image.org/docs/dev/api/skimage.measure.html

        multichannel : bool, optional If True, treat the last dimension of the array as channels. Similarity calculations are done 
        independently for each channel then averaged.
        """
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                totalSSIM += SSIM(X=ori_r_channel[i], Y=adv_r_channel[i], multichannel=True)

        print('ASS:\t{:.3f}'.format(totalSSIM / cnt))
        return totalSSIM / cnt

    # 6: PSD: Perturbation Sensitivity Distance
    def avg_PSD(self):

        psd = 0
        cnt = 0

        for outer in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[outer],
                               nature_true_preds=self.labels_samples[outer]):
                cnt += 1

                image = self.nature_samples[outer]
                pert = abs(self.adv_samples[outer] - self.nature_samples[outer])
                # my patch
                image = np.transpose(image, (1, 2, 0))
                pert = np.transpose(pert, (1, 2, 0))

                for idx_channel in range(image.shape[0]):
                    image_channel = image[idx_channel]
                    pert_channel = pert[idx_channel]

                    image_channel = np.pad(image_channel, 1, 'reflect')
                    pert_channel = np.pad(pert_channel, 1, 'reflect')

                    for i in range(1, image_channel.shape[0] - 1):
                        for j in range(1, image_channel.shape[1] - 1):
                            psd += pert_channel[i, j] * (1.0 - np.std(np.array(
                                [image_channel[i - 1, j - 1], image_channel[i - 1, j], image_channel[i - 1, j + 1],
                                 image_channel[i, j - 1],
                                 image_channel[i, j], image_channel[i, j + 1], image_channel[i + 1, j - 1],
                                 image_channel[i + 1, j],
                                 image_channel[i + 1, j + 1]])))
        print('PSD:\t{:.3f}'.format(psd / cnt))
        return psd / cnt

    # 7 NTE: Noise Tolerance Estimation
    def avg_noise_tolerance_estimation(self):

        nte = 0
        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i]):
                cnt += 1
                sort_preds = np.sort(self.softmax_prediction[i])
                nte += sort_preds[-1] - sort_preds[-2]

        print('NTE:\t{:.3f}'.format(nte / cnt))
        return nte / cnt

    # 8 RGB: Robustness to Gaussian Blur
    def robust_gaussian_blur(self, radius=0.5):

        total = 0
        num_gb = 0

        for i in range(len(self.adv_samples)):
            if np.argmax(self.softmax_prediction[i]) != np.argmax(self.labels_samples[i]):
                total += 1
                adv_sample = self.adv_samples[i]
                gb_sample = gaussian_blur_transform(AdvSample=adv_sample, radius=radius)
                gb_pred = self.model.predict(np.expand_dims(np.array(gb_sample), axis=0))
                if np.argmax(gb_pred) != np.argmax(self.labels_samples[i]):
                    num_gb += 1

        print('RGB:\t{:.3f}'.format(num_gb / total))
        return num_gb, total, num_gb / total

    # 9 RIC: Robustness to Image Compression
    def robust_image_compression(self, quality=50):

        total = 0
        num_ic = 0

        # prepare the save dir for the generated image(png or jpg)
        image_save = os.path.join('./tmp', 'image')
        if os.path.exists(image_save):
            shutil.rmtree(image_save)
        os.mkdir(image_save)
        # print('\nNow, all adversarial examples are saved as PNG and then compressed using *Guetzli* in the {} fold ......\n'.format(image_save))

        for i in range(len(self.adv_samples)):
            if np.argmax(self.softmax_prediction[i]) != np.argmax(self.labels_samples[i]):
                total += 1
                adv_sample = self.adv_samples[i]
                ic_sample = image_compress_transform(IndexAdv=i, AdvSample=adv_sample, dir_name=image_save,
                                                     quality=quality)
                ic_sample = np.expand_dims(ic_sample, axis=0)
                ic_pred = self.model.predict(np.array(ic_sample))
                if np.argmax(ic_pred) != np.argmax(self.labels_samples[i]):
                    num_ic += 1
        print('RIC:\t{:.3f}'.format(num_ic / total))
        return num_ic, total, num_ic / total

    def all(self):
        self.misclassification_rate()
        self.avg_confidence_adv_class()
        self.avg_confidence_true_class()
        self.avg_lp_distortion()
        self.avg_SSIM()
        self.avg_PSD()
        self.avg_noise_tolerance_estimation()
        self.robust_gaussian_blur()
        self.robust_image_compression(1)

# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('./data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('./data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('./data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('./data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test

