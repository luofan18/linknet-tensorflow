
# coding: utf-8

from scipy.misc import imread
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from get_class_weights import ENet_weighing, median_frequency_balancing
import augmentation
import viewer
from labels import trainId2label
from load_pretrained_weights import load_pretrained_weights

import os
import math
from tqdm import tqdm
import logging

tf.logging.set_verbosity(tf.logging.INFO)


# In[ ]:


tmp_dir = 'segmentation'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    print ('Successfully create directory.')
else: 
    print ('Directory exists.')
    
logging.basicConfig(
    filename=os.path.join(tmp_dir, 'training.log'), 
    format='%(asctime)s %(message)s', level=logging.INFO)


# In[ ]:


logging.info('Haha')


# In[ ]:


# Class index with 0~18, 19 represents others
num_class = 20
weighting = 'ENet'
combine_dataset = False

tf_logdir = './log'

train_dir = '../data/leftImg8bit/train'
val_dir = '../data/leftImg8bit/val'
test_dir = '../data/leftImg8bit/test'

gt_train_dir = '../data/gtFine/train'
gt_val_dir = '../data/gtFine/val'
gt_test_dir = '../data/gtFine/test'


# In[ ]:


def walk_dir(root):
    file_list = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_list.append(os.path.join(path, name))
    return file_list


# In[ ]:


image_train_files = walk_dir(train_dir)
image_train_files = sorted(image_train_files)
gt_train_files = walk_dir(gt_train_dir)
gt_train_files = [gt_train_file for gt_train_file in gt_train_files             if gt_train_file.endswith('_labelTrainIds.png')]
gt_train_files = sorted(gt_train_files)
print ('Read {} training images.'.format(len(image_train_files)))
print ('Read {} training groudtruth lables'.format(len(gt_train_files)))
assert len(image_train_files) == len(gt_train_files)

image_val_files = walk_dir(val_dir)
image_val_files = sorted(image_val_files)
gt_val_files = walk_dir(gt_val_dir)
gt_val_files = [gt_val_file for gt_val_file in gt_val_files                 if gt_val_file.endswith('_labelTrainIds.png')]
gt_val_files = sorted(gt_val_files)
print ('Read {} validation images.'.format(len(image_val_files)))
print ('Read {} validation groudtruth lables.'.format(len(gt_val_files)))
assert len(image_val_files) == len(gt_val_files)


# In[ ]:


if combine_dataset:
    image_train_files += image_val_files
    gt_train_files += gt_val_files


# In[ ]:


if weighting == 'MFB':
    class_weights = median_frequency_balancing(
        gt_train_files, num_classes=num_class)
    print ('Median_frequency_balancing class weights is')
    print (class_weights)
elif weighting == 'ENet':
    if os.path.isfile('enet_class_weights.npy'):
        class_weights = np.load('enet_class_weights.npy')
    else:
        class_weights = ENet_weighing(
            gt_train_files, num_classes=num_class)
        np.save('enet_class_weights.npy', class_weights)
        print ('Save Enet_class_weights')
        print ('ENet class weights is')
        
class_weights[19] = 0
print ('Id            Class                Weight')
for i in range(int(class_weights.shape[0])):
    print ('{:2}            {:15}  {:>10.6}'.
          format(i, trainId2label[i].name, class_weights[i]))


# In[ ]:


def input_pipeline(image_train_files, gt_train_files, 
                   image_val_files, gt_val_files, image_size):

    num_worker = 8
    def _parse_function(image_file, gt_file, image_size=image_size):
        image_string = tf.read_file(image_file)
        gt_string = tf.read_file(gt_file)

        image = tf.image.decode_image(image_string)
        # Need to set shape
        # https://github.com/tensorflow/tensorflow/issues/8551
        image.set_shape(shape=(1024, 2048, 3))
        image = tf.image.resize_images(image, image_size)
        #image = tf.image.per_image_standardization(image)

        gt = tf.image.decode_image(gt_string)
        gt.set_shape(shape=(1024, 2048, 1))
        gt = tf.image.resize_images(
            gt, image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        image, gt = augmentation.random_flip_left_right(image, gt)
        
        image = image -             np.array([103.939, 116.779, 123.68], dtype=np.float32)

        return image, gt

    image_train = tf.data.Dataset.from_tensor_slices(
        tf.constant(image_train_files))
    gt_train = tf.data.Dataset.from_tensor_slices(
        tf.constant(gt_train_files))
    train_dataset = tf.data.Dataset.zip((image_train, gt_train))
    # https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs
    train_dataset = train_dataset.shuffle(len(image_train_files))
    train_dataset = train_dataset.map(
        _parse_function, num_parallel_calls=num_worker
    )
    train_dataset = train_dataset.repeat()
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(image_val_files), tf.constant(gt_val_files)))
    val_dataset = val_dataset.map(
        _parse_function, num_parallel_calls=num_worker
    )
    
    return train_dataset, val_dataset


# In[ ]:


def train_step(sess, net, max_step, learning_rate, print_every=0):
    
    loss_list = []
    accuracy_list = []
    mean_iou_list = []
    
    # Training step
    for step in tqdm(range(max_step)):
        try:
            # Reset metrics counters
            sess.run(tf.local_variables_initializer())
            training_loss, _, _ = sess.run(
                [net.loss, net.metrics_update, net.opt], 
                feed_dict={
                    net.input_learning_rate: learning_rate
                })
            accuracy, mean_iou = sess.run(
                [net.accuracy, net.mean_iou]
            )
            loss_list.append(training_loss)
            accuracy_list.append(accuracy)
            mean_iou_list.append(mean_iou)
            # For print
            print_i = 0
            if print_every > 0:
                print_i = print_i + 1
                if print_i % print_every == 0:
                    print_i = 0
                    print ('Training loss is {}, accuracy is {}, mean iou is {}'                           .format(training_loss, accuracy, mean_iou))
        except tf.errors.OutOfRangeError:
            print ("Training iterator emptied.")
            break    
    print ('Mean loss in epoch is {}, accuracy is {}, mean iou is {}'.format(
        np.mean(loss_list), np.mean(accuracy_list), np.mean(mean_iou_list)
    ))
    return loss_list, accuracy_list, mean_iou_list


# In[ ]:


def val_step(sess, net, max_step, val_init):
    val_loss_list = []
    # Validation step
    # Reset validation dateset
    sess.run(val_init)
    # Reset matrics counters
    sess.run(tf.local_variables_initializer())
    val_step = 0
    for step in tqdm(range(max_step)):
        try:
            val_loss, _ = sess.run(
                [net.val_loss, net.val_metrics_update]
            )
            val_loss_list.append(val_loss)
        except tf.errors.OutOfRangeError:
            print ('Validation iterator emptied.')
    # Aggregate final results
    accuracy, mean_iou = sess.run(
        [net.val_accuracy, net.val_mean_iou]
    )
    print ('Validation accuracy is {}, mean iou is {}'.format(
        accuracy, mean_iou
    ))
    return val_loss_list, accuracy, mean_iou


# In[ ]:


def logging_header():
    formats = '{:>8.6}' + '{:>10.8}  ' * 5 + '{}'
    logging.info(formats.format(
        'epoch', 'loss', 'accuracy', 'mean_iou',
        'val_loss', 'val_acc', 'val_mean_iou'))


# In[ ]:


from train import LinkNet

label_channels = 20
input_dims = [512, 1024]
batch_size = 16

learning_rate = 0.1
learning_rate_decay = 0.1
lr_decay_every = 60
tf.reset_default_graph()

restore_training = False
start_epoch = 0
epochs = 180
train_steps = int(math.ceil(2975 / batch_size))
val_steps = int(math.ceil(500 / batch_size / 1.5))

# Get batch data
with tf.name_scope('input_pipe_line'):
    with tf.device('/cpu:0'):
        train_dataset, val_dataset = input_pipeline(
            image_train_files, gt_train_files, image_val_files, 
            gt_val_files, input_dims
        )

        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)
        train_iterator = train_dataset.make_one_shot_iterator()
        train_batch = train_iterator.get_next()

        val_dataset = val_dataset.batch(int(batch_size * 1.5))
        val_dataset = val_dataset.prefetch(1)
        val_iterator = val_dataset.make_initializable_iterator()
        val_init = val_iterator.initializer
        val_batch = val_iterator.get_next()

net = LinkNet(
    input_dims, label_channels, class_weights, 
    input_method='dataset_api', inputs=train_batch, 
    inputs_val=val_batch)

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.logging.set_verbosity(tf.logging.INFO)
graph_writer = tf.summary.FileWriter('log/train', sess.graph)

if restore_training:
    net.restore(sess, os.path.join(tmp_dir, 
        'model-e{}.ckpt'.format(start_epoch - 1)))
    train_metrics = np.load(
        os.path.join(tmp_dir, 'train_metrics-e{}.npy'.\
                     format(start_epoch - 1))).item()
    val_metrics = np.load(
        os.path.join(tmp_dir, 'train_metrics-e{}.npy'.\
                     format(start_epoch - 1))).item()
else:
    load_pretrained_weights
    train_metrics = {}
    val_metrics = {}

metrics = ['loss', 'accuracy', 'mean_iou']
def store_metrics(dictionary, metrics, values):
    for metric, value in zip(metrics, values):
        dictionary.setdefault(metric, []).append(value)

logging_header()
for epoch in range(epochs):
    if start_epoch + epoch != 0 and (start_epoch + epoch) % lr_decay_every == 0:
        learning_rate = learning_rate * learning_rate_decay
        logging.info(
            'Learning rate change to {}.'.format(learning_rate)
        )
    print ('Epoch {} begin.'.format(start_epoch + epoch))
    loss_list, accuracy_list, mean_iou_list = train_step(
        sess, net, train_steps, learning_rate)
    val_loss_list, val_accuracy, val_mean_iou = val_step(
        sess, net, val_steps, val_init)
    
    store_metrics(
        train_metrics, metrics, [np.mean(loss_list), 
        np.mean(accuracy_list), np.mean(mean_iou_list)]
    )
    store_metrics(
        val_metrics, metrics, [np.mean(val_loss_list),
        val_accuracy, val_mean_iou]
    )
    net.save(sess, os.path.join(
        tmp_dir, 'model-e{}.ckpt'.format(start_epoch + epoch)))
    
    formats = '{:8d}' + '{:>10.6}  ' * 6
    logging.info(formats.format(
        start_epoch + epoch, np.mean(loss_list), 
        np.mean(accuracy_list), np.mean(mean_iou_list), 
        np.mean(val_loss_list), val_accuracy, val_mean_iou))
    np.save(os.path.join(tmp_dir, 
            'train_metrics-e{}.npy'.format(start_epoch + epoch)), 
            np.array(train_metrics))
    np.save(os.path.join(tmp_dir, 
            'train_metrics-e{}.npy'.format(start_epoch, epoch)),
            np.array(val_metrics))

