from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

import os

from labels import trainId2label


def overlap(image, label, alpha=0.5):

    label = label2color(label)

    image = image * (1 - alpha) + label * alpha
    image = np.rint(image)
    return image


def label2color(label):
    h, w = label.shape
    label = label.flatten()
    label = np.expand_dims(label, 1)

    def id2label(id):
        return np.array(trainId2label[int(id)].color)

    color = np.apply_along_axis(id2label, -1, label)
    color = np.resize(color, (h, w, 3))
    color = color.astype('uint8')
    return color


def predict_and_show(num, image_file_names, gt_file_names, sess, net, save_dir):
    for i in range(num):
        id = np.random.randint(len(image_file_names))
        images = imread(image_file_names[id])
        pred = net.predict(sess, images)
        pred = label2color(pred[0])
        gt = imread(gt_file_names[id])
        gt = label2color(gt)
        show_3_images(images, pred, gt, (16, 4),
                             os.path.join(save_dir,
                                          image_file_names[id].split('/')[-1]))


def show_2_images(i1, i2, figsize=None):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(i1)
    plt.subplot(122)
    plt.imshow(i2)
    plt.show()


def show_3_images(i1, i2, i3, figsize=None, save_to=None):
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(i1)
    plt.subplot(132)
    plt.imshow(i2)
    plt.subplot(133)
    plt.imshow(i3)
    plt.savefig(save_to, dpi=300)
    plt.show()


def show_4_images(i1, i2, i3, i4, figsize=None):
    plt.figure(figsize=figsize)
    plt.subplot(141)
    plt.imshow(i1)
    plt.subplot(142)
    plt.imshow(i2)
    plt.subplot(143)
    plt.imshow(i3)
    plt.subplot(144)
    plt.imshow(i4)
    plt.show()