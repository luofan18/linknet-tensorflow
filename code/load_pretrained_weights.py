import os
import tensorflow as tf
import numpy as np
import torchfile  # pip install torchfile

#From https://github.com/dalgu90/resnet-18-tensorflow/blob/master/extract_torch_t7.py

T7_PATH = './resnet-18.t7'


def load_pretrained_weights(sess, T7_PATH=T7_PATH):
    # Open ResNet-18 torch checkpoint
    print('Open ResNet-18 torch checkpoint: %s' % T7_PATH)
    o = torchfile.load(T7_PATH)

    # Load weights in a brute-force way
    print('Load weights in a brute-force way')
    conv1_weights = o.modules[0].weight
    conv1_bn_gamma = o.modules[1].weight
    conv1_bn_beta = o.modules[1].bias
    conv1_bn_mean = o.modules[1].running_mean
    conv1_bn_var = o.modules[1].running_var

    conv2_1_weights_1  = o.modules[4].modules[0].modules[0].modules[0].modules[0].weight
    conv2_1_bn_1_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[1].weight
    conv2_1_bn_1_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[1].bias
    conv2_1_bn_1_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_mean
    conv2_1_bn_1_var   = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_var
    conv2_1_weights_2  = o.modules[4].modules[0].modules[0].modules[0].modules[3].weight
    conv2_1_bn_2_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[4].weight
    conv2_1_bn_2_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[4].bias
    conv2_1_bn_2_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_mean
    conv2_1_bn_2_var   = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_var
    conv2_2_weights_1  = o.modules[4].modules[1].modules[0].modules[0].modules[0].weight
    conv2_2_bn_1_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[1].weight
    conv2_2_bn_1_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[1].bias
    conv2_2_bn_1_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_mean
    conv2_2_bn_1_var   = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_var
    conv2_2_weights_2  = o.modules[4].modules[1].modules[0].modules[0].modules[3].weight
    conv2_2_bn_2_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[4].weight
    conv2_2_bn_2_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[4].bias
    conv2_2_bn_2_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_mean
    conv2_2_bn_2_var   = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_var

    conv3_1_weights_skip = o.modules[5].modules[0].modules[0].modules[1].weight
    conv3_1_weights_1  = o.modules[5].modules[0].modules[0].modules[0].modules[0].weight
    conv3_1_bn_1_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[1].weight
    conv3_1_bn_1_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[1].bias
    conv3_1_bn_1_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_mean
    conv3_1_bn_1_var   = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_var
    conv3_1_weights_2  = o.modules[5].modules[0].modules[0].modules[0].modules[3].weight
    conv3_1_bn_2_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[4].weight
    conv3_1_bn_2_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[4].bias
    conv3_1_bn_2_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_mean
    conv3_1_bn_2_var   = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_var
    conv3_2_weights_1  = o.modules[5].modules[1].modules[0].modules[0].modules[0].weight
    conv3_2_bn_1_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[1].weight
    conv3_2_bn_1_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[1].bias
    conv3_2_bn_1_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_mean
    conv3_2_bn_1_var   = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_var
    conv3_2_weights_2  = o.modules[5].modules[1].modules[0].modules[0].modules[3].weight
    conv3_2_bn_2_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[4].weight
    conv3_2_bn_2_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[4].bias
    conv3_2_bn_2_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_mean
    conv3_2_bn_2_var   = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_var

    conv4_1_weights_skip = o.modules[6].modules[0].modules[0].modules[1].weight
    conv4_1_weights_1  = o.modules[6].modules[0].modules[0].modules[0].modules[0].weight
    conv4_1_bn_1_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[1].weight
    conv4_1_bn_1_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[1].bias
    conv4_1_bn_1_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_mean
    conv4_1_bn_1_var   = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_var
    conv4_1_weights_2  = o.modules[6].modules[0].modules[0].modules[0].modules[3].weight
    conv4_1_bn_2_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[4].weight
    conv4_1_bn_2_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[4].bias
    conv4_1_bn_2_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_mean
    conv4_1_bn_2_var   = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_var
    conv4_2_weights_1  = o.modules[6].modules[1].modules[0].modules[0].modules[0].weight
    conv4_2_bn_1_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[1].weight
    conv4_2_bn_1_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[1].bias
    conv4_2_bn_1_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_mean
    conv4_2_bn_1_var   = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_var
    conv4_2_weights_2  = o.modules[6].modules[1].modules[0].modules[0].modules[3].weight
    conv4_2_bn_2_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[4].weight
    conv4_2_bn_2_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[4].bias
    conv4_2_bn_2_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_mean
    conv4_2_bn_2_var   = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_var

    conv5_1_weights_skip = o.modules[7].modules[0].modules[0].modules[1].weight
    conv5_1_weights_1  = o.modules[7].modules[0].modules[0].modules[0].modules[0].weight
    conv5_1_bn_1_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[1].weight
    conv5_1_bn_1_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[1].bias
    conv5_1_bn_1_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_mean
    conv5_1_bn_1_var   = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_var
    conv5_1_weights_2  = o.modules[7].modules[0].modules[0].modules[0].modules[3].weight
    conv5_1_bn_2_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[4].weight
    conv5_1_bn_2_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[4].bias
    conv5_1_bn_2_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_mean
    conv5_1_bn_2_var   = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_var
    conv5_2_weights_1  = o.modules[7].modules[1].modules[0].modules[0].modules[0].weight
    conv5_2_bn_1_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[1].weight
    conv5_2_bn_1_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[1].bias
    conv5_2_bn_1_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_mean
    conv5_2_bn_1_var   = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_var
    conv5_2_weights_2  = o.modules[7].modules[1].modules[0].modules[0].modules[3].weight
    conv5_2_bn_2_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[4].weight
    conv5_2_bn_2_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[4].bias
    conv5_2_bn_2_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_mean
    conv5_2_bn_2_var   = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_var

    fc_weights = o.modules[10].weight
    fc_biases = o.modules[10].bias

    model_weights_map = {
        'initial_block/conv1/kernel': conv1_weights,
        'initial_block/bn1/moving_mean': conv1_bn_mean,
        'initial_block/bn1/moving_variance': conv1_bn_var,
        'initial_block/bn1/beta': conv1_bn_beta,
        'initial_block/bn1/gamma': conv1_bn_gamma,

        'eb1/bb1/conv_bn_relu/conv/kernel': conv2_1_weights_1,
        'eb1/bb1/conv_bn_relu/bn/moving_mean': conv2_1_bn_1_mean,
        'eb1/bb1/conv_bn_relu/bn/moving_variance': conv2_1_bn_1_var,
        'eb1/bb1/conv_bn_relu/bn/beta': conv2_1_bn_1_beta,
        'eb1/bb1/conv_bn_relu/bn/gamma': conv2_1_bn_1_gamma,
        'eb1/bb1/conv2/kernel': conv2_1_weights_2,
        'eb1/bb1/bn2/moving_mean': conv2_1_bn_2_mean,
        'eb1/bb1/bn2/moving_variance': conv2_1_bn_2_var,
        'eb1/bb1/bn2/beta': conv2_1_bn_2_beta,
        'eb1/bb1/bn2/gamma': conv2_1_bn_2_gamma,
        'eb1/bb2/conv_bn_relu/conv/kernel': conv2_2_weights_1,
        'eb1/bb2/conv_bn_relu/bn/moving_mean': conv2_2_bn_1_mean,
        'eb1/bb2/conv_bn_relu/bn/moving_variance': conv2_2_bn_1_var,
        'eb1/bb2/conv_bn_relu/bn/beta': conv2_2_bn_1_beta,
        'eb1/bb2/conv_bn_relu/bn/gamma': conv2_2_bn_1_gamma,
        'eb1/bb2/conv2/kernel': conv2_2_weights_2,
        'eb1/bb2/bn2/moving_mean': conv2_2_bn_2_mean,
        'eb1/bb2/bn2/moving_variance': conv2_2_bn_2_var,
        'eb1/bb2/bn2/beta': conv2_2_bn_2_beta,
        'eb1/bb2/bn2/gamma': conv2_2_bn_2_gamma,

        'eb2/bb1/dimension_reduction/kernel': conv3_1_weights_skip,
        'eb2/bb1/conv_bn_relu/conv/kernel': conv3_1_weights_1,
        'eb2/bb1/conv_bn_relu/bn/moving_mean': conv3_1_bn_1_mean,
        'eb2/bb1/conv_bn_relu/bn/moving_variance': conv3_1_bn_1_var,
        'eb2/bb1/conv_bn_relu/bn/beta': conv3_1_bn_1_beta,
        'eb2/bb1/conv_bn_relu/bn/gamma': conv3_1_bn_1_gamma,
        'eb2/bb1/conv2/kernel': conv3_1_weights_2,
        'eb2/bb1/bn2/moving_mean': conv3_1_bn_2_mean,
        'eb2/bb1/bn2/moving_variance': conv3_1_bn_2_var,
        'eb2/bb1/bn2/beta': conv3_1_bn_2_beta,
        'eb2/bb1/bn2/gamma': conv3_1_bn_2_gamma,
        'eb2/bb2/conv_bn_relu/conv/kernel': conv3_2_weights_1,
        'eb2/bb2/conv_bn_relu/bn/moving_mean': conv3_2_bn_1_mean,
        'eb2/bb2/conv_bn_relu/bn/moving_variance': conv3_2_bn_1_var,
        'eb2/bb2/conv_bn_relu/bn/beta': conv3_2_bn_1_beta,
        'eb2/bb2/conv_bn_relu/bn/gamma': conv3_2_bn_1_gamma,
        'eb2/bb2/conv2/kernel': conv3_2_weights_2,
        'eb2/bb2/bn2/moving_mean': conv3_2_bn_2_mean,
        'eb2/bb2/bn2/moving_variance': conv3_2_bn_2_var,
        'eb2/bb2/bn2/beta': conv3_2_bn_2_beta,
        'eb2/bb2/bn2/gamma': conv3_2_bn_2_gamma,

        'eb3/bb1/dimension_reduction/kernel': conv4_1_weights_skip,
        'eb3/bb1/conv_bn_relu/conv/kernel': conv4_1_weights_1,
        'eb3/bb1/conv_bn_relu/bn/moving_mean': conv4_1_bn_1_mean,
        'eb3/bb1/conv_bn_relu/bn/moving_variance': conv4_1_bn_1_var,
        'eb3/bb1/conv_bn_relu/bn/beta': conv4_1_bn_1_beta,
        'eb3/bb1/conv_bn_relu/bn/gamma': conv4_1_bn_1_gamma,
        'eb3/bb1/conv2/kernel': conv4_1_weights_2,
        'eb3/bb1/bn2/moving_mean': conv4_1_bn_2_mean,
        'eb3/bb1/bn2/moving_variance': conv4_1_bn_2_var,
        'eb3/bb1/bn2/beta': conv4_1_bn_2_beta,
        'eb3/bb1/bn2/gamma': conv4_1_bn_2_gamma,
        'eb3/bb2/conv_bn_relu/conv/kernel': conv4_2_weights_1,
        'eb3/bb2/conv_bn_relu/bn/moving_mean': conv4_2_bn_1_mean,
        'eb3/bb2/conv_bn_relu/bn/moving_variance': conv4_2_bn_1_var,
        'eb3/bb2/conv_bn_relu/bn/beta': conv4_2_bn_1_beta,
        'eb3/bb2/conv_bn_relu/bn/gamma': conv4_2_bn_1_gamma,
        'eb3/bb2/conv2/kernel': conv4_2_weights_2,
        'eb3/bb2/bn2/moving_mean': conv4_2_bn_2_mean,
        'eb3/bb2/bn2/moving_variance': conv4_2_bn_2_var,
        'eb3/bb2/bn2/beta': conv4_2_bn_2_beta,
        'eb3/bb2/bn2/gamma': conv4_2_bn_2_gamma,

        'eb4/bb1/dimension_reduction/kernel': conv5_1_weights_skip,
        'eb4/bb1/conv_bn_relu/conv/kernel': conv5_1_weights_1,
        'eb4/bb1/conv_bn_relu/bn': conv5_1_bn_1_mean,
        'eb4/bb1/conv_bn_relu/bn/moving_variance': conv5_1_bn_1_var,
        'eb4/bb1/conv_bn_relu/bn/beta': conv5_1_bn_1_beta,
        'eb4/bb1/conv_bn_relu/bn/gamma': conv5_1_bn_1_gamma,
        'eb4/bb1/conv2/kernel': conv5_1_weights_2,
        'eb4/bb1/bn2/moving_mean': conv5_1_bn_2_mean,
        'eb4/bb1/bn2/moving_variance': conv5_1_bn_2_var,
        'eb4/bb1/bn2/beta': conv5_1_bn_2_beta,
        'eb4/bb1/bn2/gamma': conv5_1_bn_2_gamma,
        'eb4/bb2/conv_bn_relu/conv/kernel': conv5_2_weights_1,
        'eb4/bb2/conv_bn_relu/bn/moving_mean': conv5_2_bn_1_mean,
        'eb4/bb2/conv_bn_relu/bn/moving_variance': conv5_2_bn_1_var,
        'eb4/bb2/conv_bn_relu/bn/beta': conv5_2_bn_1_beta,
        'eb4/bb2/conv_bn_relu/bn/gamma': conv5_2_bn_1_gamma,
        'eb4/bb2/conv2/kernel': conv5_2_weights_2,
        'eb4/bb2/bn2/moving_mean': conv5_2_bn_2_mean,
        'eb4/bb2/bn2/moving_variance': conv5_2_bn_2_var,
        'eb4/bb2/bn2/beta': conv5_2_bn_2_beta,
        'eb4/bb2/bn2/gamma': conv5_2_bn_2_gamma,

    #    'logits/fc/weights': fc_weights,
    #    'logits/fc/biases': fc_biases,
    }

    # Transpose conv and fc weights
    model_weights = {}
    for k, v in model_weights_map.items():
        if len(v.shape) == 4:
            model_weights[k] = np.transpose(v, (2, 3, 1, 0))
        elif len(v.shape) == 2:
            model_weights[k] = np.transpose(v)
        else:
            model_weights[k] = v

    with tf.variable_scope('linknet', reuse=True):
        for k, v in model_weights_map.items():
            sess.run(tf.get_variable(k).assign(v))