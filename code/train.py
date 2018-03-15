import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imresize
from linknet import linknet
from metrics import dice_coef


def model_inputs(image_height, image_width, image_channels):
    return (
        tf.placeholder(tf.float32,
                       (None, image_height, image_width, image_channels),
                       name='inputs'),
        tf.placeholder(tf.float32,
                       (None, image_height, image_width, 1),
                       name='labels'),
        tf.placeholder(tf.float32, name='learning_rate'),
        tf.placeholder(tf.bool, name='is_training')
    )


def model_inputs_with_default(image_height, image_width, image_channels,
                              inputs_images, inputs_labels,
                              inputs_val_images=None,
                              inputs_val_labels=None):
    if inputs_val_images is None:
        return (
            tf.placeholder_with_default(inputs_images,
                                        (None, image_height, image_width,
                                         image_channels),
                                        name='inputs'),
            tf.placeholder_with_default(inputs_labels,
                                        (None, image_height, image_width,
                                         1),
                                        name='labels'),
            tf.placeholder(tf.float32, name='learning_rate'),
            tf.placeholder(tf.bool, name='is_training')
        )
    else:
        # Input handle to select whether use training data or validation
        # data. 1 for training data, 0 for validation data.
        inputs_handle = tf.placeholder(tf.bool, name='inputs_handle')
        inputs_images = tf.cond(inputs_handle, lambda: inputs_images,
                                lambda: inputs_val_images,
                                name='inputs_images_handle'
        )
        tf.add_to_collection('handle', inputs_images)
        inputs_labels = tf.cond(inputs_handle, lambda: inputs_labels,
                                lambda: inputs_val_labels,
                                name='inputs_labels_handle')
        tf.add_to_collection('handle', inputs_labels)
        return (
            tf.placeholder_with_default(inputs_images,
                                        (None, image_height, image_width,
                                         image_channels),
                                        name='inputs'),
            tf.placeholder_with_default(inputs_labels,
                                        (None, image_height, image_width,
                                         1),
                                        name='labels'),
            tf.placeholder(tf.float32, name='learning_rate'),
            tf.placeholder(tf.bool, name='is_training'),
            inputs_handle
        )


def model_loss(logits, onehot_labels,  class_weights=[1, ], weight_decay=2e-4):
    if len(class_weights) > 1:
        weights = onehot_labels * class_weights
        weights = tf.reduce_sum(weights, 3)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                 logits=logits,
                                                 weights=weights)

    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=onehot_labels, logits=logits
        )
        loss = tf.reduce_mean(loss)

    # Add regularization loss
    # https://github.com/tensorflow/models/blob/master/official/resnet/cifar10_main.py#L184
    # Include batch normalization params
    vars = tf.trainable_variables()
    vars = [var for var in vars if var.name.startswith('linknet')]
    loss = loss + weight_decay * tf.add_n(
        [tf.nn.l2_loss(var) for var in vars]
    )

    return loss


def model_opt(loss, learning_rate,):

    vars = tf.trainable_variables()
    vars = [var for var in vars if var.name.startswith('linknet')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # In paper, they use RMSProp, in Pytorch code, they use SGD
        opt = tf.train.MomentumOptimizer(
            learning_rate, momentum=0.9).minimize(loss, var_list=vars)
    '''
    opt = slim.learning.create_train_op(
        loss, tf.train.GradientDescentOptimizer(learning_rate))
    '''
    return opt


class LinkNet:
    def __init__(self, input_dims, num_class, class_weights=[1, ],
                 reuse=None, input_method='feed_dict', inputs=None,
                 inputs_val=None, model_param={}):
        image_height, image_width = input_dims
        self.image_height = image_height
        self.image_width = image_width
        image_channels = 3

        self.weight_decay = model_param.get('weight_decay', 2e-4)

        if input_method is 'feed_dict':
            # Input placeholder
            self.input_images, self.input_labels, self.input_learning_rate, \
                self.is_training = model_inputs(
                    image_height, image_width, image_channels
            )
            # TODO
        elif input_method is 'dataset_api':
            inputs_images, inputs_labels = inputs

            # Training graph
            self.input_learning_rate = tf.placeholder(
                tf.float32, name='learning_rate'
            )
            self.prob, self.logits = linknet(
                inputs_images, num_class, reuse=reuse, is_training=True
            )
            # Loss
            with tf.name_scope('loss'):
                if num_class == 1:
                    self.loss = model_loss(
                        self.logits, inputs_labels, class_weights,
                        self.weight_decay)
                else:
                    onehot_labels = tf.one_hot(
                        tf.squeeze(inputs_labels, -1), num_class, axis=-1
                    )
                    self.loss = model_loss(
                        self.logits, onehot_labels, class_weights,
                        self.weight_decay)

            self.opt = model_opt(self.loss, self.input_learning_rate)

            # Metrics
            train_pred = tf.argmax(self.prob, axis=-1)
            inputs_labels = tf.squeeze(inputs_labels, -1)
            self.accuracy, self.accuracy_update = tf.metrics.accuracy(
                inputs_labels, train_pred
            )
            self.mean_iou, self.mean_iou_update = tf.metrics.mean_iou(
                inputs_labels, train_pred, num_class
            )
            self.metrics_update = tf.group(
                self.accuracy_update, self.mean_iou_update
            )

            # Validation graph
            if inputs_val is not None:
                val_images, val_labels = inputs_val
                self.val_prob, self.val_logits = linknet(
                    val_images, num_class, reuse=True, is_training=False
                )
                with tf.name_scope('val_loss'):
                    if num_class == 1:
                        # TODO
                        pass
                    else:
                        onehot_labels = tf.one_hot(
                            tf.squeeze(val_labels, -1), num_class, axis=-1
                        )
                        self.val_loss = model_loss(
                            self.val_logits, onehot_labels, class_weights
                        )

                val_pred = tf.argmax(self.val_prob, axis=-1)
                val_labels = tf.squeeze(val_labels, -1)
                self.val_accuracy, self.val_accuracy_update = \
                    tf.metrics.accuracy(
                        val_labels, val_pred
                    )
                self.val_mean_iou, self.val_mean_iou_update = \
                    tf.metrics.mean_iou(
                        val_labels, val_pred, num_class
                    )
                self.val_metrics_update = tf.group(
                    self.val_accuracy_update, self.val_mean_iou_update
                )

            # Prediction graph with placeholder
            self.input_images = tf.placeholder(
                tf.float32, (None, image_height, image_width, image_channels),
                name='prediction_input_images'
            )
            input_images = tf.map_fn(
                tf.image.per_image_standardization, self.input_images
            )
            self.pred_prob, self.pred_logits = linknet(
                input_images, num_class, reuse=True, is_training=False
            )
            self.pred = tf.argmax(self.pred_prob, -1)

        # Prepare for saving model
        self.saver = tf.train.Saver()
        self.save_path = []

    def predict(self, sess, images):
        # TODO support batches

        images = imresize(images, (self.image_height, self.image_width))
        images = np.expand_dims(images, 0)
        return sess.run(self.pred, feed_dict={
            self.input_images: images
        })

    def save(self, sess, save_path):
        self.save_path = self.saver.save(sess, save_path)
        print ('Model saved in path: {}'.format(save_path))

    def restore(self, sess, save_path):
        self.saver.restore(sess, save_path)
        print ('Model restored.')