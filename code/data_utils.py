import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import (
    array_to_img, img_to_array, load_img, ImageDataGenerator)

import pdb
import random
from PIL import Image
import numpy as np

def random_horizontal_flip(image_pair, prob=0.5):
    """
    image: PIL image
    label: PIL image
    prob: the probability the image is flipped
    Note: wired behavior for transpose, result in NoneType error 
    when use lambda and list
    """
    image, label = image_pair
    image = array_to_img(image)
    w, h = image.size
    flip_matrix = (-1, 0, w, 0, 1, 0)
    if random.random() < prob:
        image = image.transform(image.size, Image.AFFINE, flip_matrix)
        label = label.transform(image.size, Image.AFFINE, flip_matrix)
        assert image is not None and label is not None
    return image, label

def random_rotate(image_pair, rotate_limit=(-30, 30)):
    image, label = image_pair
    rotate = random.uniform(rotate_limit[0], rotate_limit[1])
    image = image.rotate(rotate)
    label = label.rotate(rotate)
    return image, label

def shift_image(image, x, y):
    # https://stackoverflow.com/questions/17056209/python-pil-affine-transformation
    data = (1, 0, -x, 0, 1, -y)
    return image.transform(image.size, Image.AFFINE, data)

def random_scale_shift(image_pair, shift_limit=(-0.0625, 0.0625), 
                scale_limit=(-0.1, 0.1)):
    """
    the output image size may be different from the input
    """
    image, label = image_pair
    shift_x = random.uniform(shift_limit[0], shift_limit[1])
    shift_y = random.uniform(shift_limit[0], shift_limit[1])
    scale = random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    
    w, h = image.size
    
    image = image.resize((round(w * scale), round(h * scale)), resample=Image.BILINEAR)
    label = label.resize((round(w * scale), round(h * scale)), resample=Image.NEAREST)
    
    # shift
    w, h = image.size
    image = shift_image(image, shift_x * w, shift_y * h)
    lable = shift_image(label, shift_x * w, shift_y * h)
    return image, label


def random_crop_along_width(image_pair, dims):
    image, mask = image_pair
    assert dims[0] == 1280, "wrong dims"
    x_s = random.randint(0, 1918 - dims[1])
    box = (x_s, 0, x_s+dims[1], dims[0])
    image = image.crop(box)
    mask = mask.crop(box)
    return image, mask

def random_crop(image_pair, dims):
    image, mask = image_pair
    w, h = image.size
    x_s = random.randint(0, w - dims[1] + 1)
    y_s = random.randint(0, h - dims[0] + 1)
    box = (x_s, y_s, x_s+dims[1], y_s+dims[0])
    image = image.crop(box)
    mask = mask.crop(box)
    return image, mask

def cut_mask(image_pair, dims):
    assert dims[0] % 2 == 0 and dims[1] % 2 == 0, 'Output dims must be even numbers.'
    image, mask = image_pair
    # Because the image size is reversed of what we defined
    original_dims = image.size[::-1]
    x_s = (original_dims[1] - dims[1]) / 2
    y_s = (original_dims[0] - dims[0]) / 2
    box = (x_s, y_s, x_s+dims[1], y_s+dims[0])
    mask = mask.crop(box)
    return image, mask

def resize_image(image_pair, dims):
    image, mask = image_pair
    image = image.resize(dims, resample=Image.BILINEAR)
    mask = mask.resize(dims, resample=Image.BILINEAR)
    return image, mask

def pad_image_and_mask(image_pair, pad_width, mode='symmetric'):
    """
    Padding the image for prediction, the mode is 
    same as the one in numpy.pad
    
    # Args 
    image:      PIL Image or array
    pad_width:  the width to pad around the image, integer or tuple represent
                ((h, h), (w, w), (channel, channel))
    
    # Return
                PIL Image padded
    """
    image, mask = image_pair
    
    array_image = img_to_array(image)
    array_mask = img_to_array(mask)
    if type(pad_width) is int:
        padding = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    else:
        assert type(pad_width) in [tuple, list]
        padding = pad_width
    padded_array_image = np.pad(array_image, padding, mode=mode)
    padded_array_mask = np.pad(array_mask, padding, mode=mode)
    
    padded_image = array_to_img(padded_array_image)
    padded_mask = array_to_img(padded_array_mask)
    
    return padded_image, padded_mask

def apply_transforms(image, mask, transforms):
    #pdb.set_trace()
    image_pair = (image, mask)
    assert transforms is not None
    for transform in transforms:
        image, mask = transform(image_pair)
        image_pair = (image, mask)
    return image, mask

# Generator that we will use to read the data from the directory
def data_gen_small(data_dir, mask_dir, images, batch_size, dims, transforms=None, 
                   in_order=False):
    """
    data_dir:     where the actual images are kept
    mask_dir:     where the actual masks are kept
    images:       the filenames of the images we want to generate batches from
    dims:         the dimensions in which we want to rescale our images
    transforms:   the list of transforms to be applied to image and mask in order,
                  If transform is used, the image will not be resized
    in_order:     If ture, then the generator will generate the data in original 
                  order
                  
    Return:       multidimentional array, the first axis is the indice of image or
                  mask
    """
    if batch_size == 0:
        img_number = len(images)
        all_imgs = []
        all_masks = []
        print('Build generator, this may take some time...')
        print('total number of images: {}'.format(img_number))
        for i, image in enumerate(tqdm(images)):
            # Read image and mask
            original_img = load_img(data_dir + image)
            original_mask = load_img(mask_dir + image.split('.')[0] + '_mask.gif')
            
            # Apply transform
            if transforms:
                transformed_img, transformed_mask = \
                    apply_transforms(original_img, original_mask, transforms)
                resized_img, resized_mask = transformed_img, transformed_mask
            else:
                resized_img = imresize(original_img, dims+[3])
                resized_mask = imresize(original_mask, dims+[3])
            # Image
            array_img = img_to_array(resized_img) / 255
            all_imgs.append(array_img)
            # Mask
            array_mask = img_to_array(resized_mask) / 255
            all_masks.append(array_mask)
        all_imgs = np.array(all_imgs)
        all_masks = np.array(all_masks)
        while True:
            if all_masks.shape[-1] == 3:
                yield all_imgs, all_masks[:,:,:,0][:,:,:,None]
            else:
                yield all_imgs, all_masks[:,:,:,None]
            
    else:
        ix = None
        while True:
            img_number = len(images)
            if not in_order:
                # Generate random sequence of data
                ix = np.random.choice(np.arange(len(images)), batch_size)
            else:
                # Generate data in same order
                if ix is None:
                    # initialize
                    begin_i = 0
                    end_i = batch_size
                    ix = range(begin_i, end_i)
                else:
                    if end_i == len(images):
                        begin_i = 0
                    else:
                        begin_i = end_i 
                    end_i = begin_i + batch_size
                    if end_i >= len(images):
                        end_i = len(images)
                    ix = range(begin_i, end_i)
            imgs = []
            labels = []
            for i in ix:
                original_img, original_mask = \
                    read_image_and_mask(data_dir, mask_dir, images[i])
                
                # Apply transform
                if transforms:
                    transformed_img, transformed_mask = \
                        apply_transforms(original_img, original_mask, transforms)
                    resized_img, resized_mask = transformed_img, transformed_mask
                else:
                    resized_img = imresize(original_img, dims+[3])
                    resized_mask = imresize(original_mask, dims+[3])
                # images
                array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
                
                # masks
                array_mask = img_to_array(resized_mask)/255
                labels.append(array_mask[:, :, 0])
            imgs = np.array(imgs)
            labels = np.array(labels)
            if labels.shape[-1] == 3:
                yield imgs, labels[:,:,:,0][:,:,:,None]
            else:
                yield imgs, labels[:,:,:,None]
            
class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by serializing
    call to the 'next' method of given iterator/generator.
    """
    def __init__(self, it):
        import threading
        self.it = it
        self.lock = threading.Lock()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        with self.lock:
            return next(self.it)
        
# Utility function to convert greyscale images to rgb
def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j]) * 3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img

def read_image(data_dir, image):
    return load_img(data_dir + image)

def read_image_and_mask(data_dir, mask_dir, image):
    return (load_img(data_dir + image), 
            load_img(mask_dir + image.split('.')[0] + '_mask.gif'))

def show_mask_on_image(image, mask):
    if mask.ndim == 2:
        mask = mask[:,:,None]
    
    plt.figure(figsize=(16 ,12))
    plt.imshow(image, alpha=0.5)
    plt.imshow(mask[:,:,0], alpha=0.5)
    plt.show()

def show_image_and_mask(image, mask):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()

def show_2_images(i1, i2):
    plt.figure()
    plt.subplot(121)
    plt.imshow(i1)
    plt.subplot(122)
    plt.imshow(i2)
    plt.show()

def show_3_images(i1, i2, i3):
    plt.figure()
    plt.subplot(131)
    plt.imshow(i1)
    plt.subplot(132)
    plt.imshow(i2)
    plt.subplot(133)
    plt.imshow(i3)
    plt.show()
    
def show_4_images(i1, i2, i3, i4):
    plt.figure()
    plt.subplot(141)
    plt.imshow(i1)
    plt.subplot(142)
    plt.imshow(i2)
    plt.subplot(143)
    plt.imshow(i3)
    plt.subplot(144)
    plt.imshow(i4)
    plt.show()
    
def show_diff(image, mask, mask_pred, save_to=None, show=True):
    plt.figure(figsize=(20, 16 * 4))
    plt.subplot(411)
    plt.imshow(image)
    plt.subplot(412)
    plt.imshow(mask)
    plt.subplot(413)
    plt.imshow(mask_pred)
    plt.subplot(414)
    plt.imshow(mask - mask_pred)
    if save_to:
        plt.savefig(save_to)
    if show:
        plt.show()
    plt.close()