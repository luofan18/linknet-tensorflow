"""
This code is partially from
https://github.com/kwotsin/TensorFlow-ENet/blob/master/get_class_weights.py
"""
import numpy as np
import os
from scipy.misc import imread
from tqdm import tqdm
xrange = range


def ENet_weighing(image_files, num_classes=12):
    '''
    The custom class weighing function as seen in the ENet paper.

    :param image_files: a list of image_filenames
    :param num_classes:
    :return: class_weights(list): a list of class weights
    '''

    hist = np.zeros((num_classes))
    for image_file in tqdm(image_files):
        image = imread(image_file)
        # list(range(num_classes)) define the bin edges
        h, bins = np.histogram(image, list(range(num_classes + 1)))
        hist += h

    hist = hist/np.max(hist)
    class_weights = 1 / np.log(1.02 + hist)

    return class_weights

def median_frequency_balancing(image_files, num_classes=12):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c

    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.

    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images

    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in xrange(num_classes):
        label_to_frequency_dict[i] = []

    for n in tqdm(xrange(len(image_files))):
        image = imread(image_files[n])

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in xrange(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
    class_weights[-1] = 0.0

    return class_weights