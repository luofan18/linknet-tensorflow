import tensorflow as tf
from tensorflow.python.keras.initializers import he_normal
from tensorflow.contrib.framework import arg_scope, add_arg_scope


@add_arg_scope
def conv_bn_relu(x, num_channel, kernel_size, stride,
                 is_training, scope, padding='same', use_bias=False):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(x, num_channel, [kernel_size, kernel_size],
                             strides=stride, activation=None, name='conv',
                             padding=padding, use_bias=use_bias,
                             kernel_initializer=he_normal())
        x = tf.layers.batch_normalization(x, momentum=0.9, training=is_training,
                                          name='bn')
        x = tf.nn.relu(x, name='relu')
        return x


@add_arg_scope
def basic_block(x, num_channel, kernel_size,
                stride, is_training, scope, padding='same'):
    # Shortcut connection
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        if in_channel == num_channel:
            if stride == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [stride, stride], stride,
                                          padding='same')
        else:
            # Considering maxpooling if stride > 1
            shortcut = tf.layers.conv2d(x, num_channel, 1, strides=stride,
                                        padding='same', activation=None,
                                        name='dimension_reduction',
                                        use_bias=False,
                                        kernel_initializer=he_normal())
            # shortcut = tf.layers.batch_normalization(
            #     shortcut, momentum=0.9, training=is_training,
            #     name='shortcut_bn'
            # )

        x = conv_bn_relu(x, num_channel, kernel_size, stride,
                         is_training=is_training, scope='conv_bn_relu',
                         padding=padding)
        x = tf.layers.conv2d(x, num_channel, [kernel_size, kernel_size],
                             strides=1, padding=padding, name='conv2',
                             use_bias=False, kernel_initializer=he_normal())
        x = tf.layers.batch_normalization(x, momentum=0.9,
                                          training=is_training, name='bn2')
        # Considering add relu to x before addition
        x = x + shortcut
        x = tf.nn.relu(x)
    return x


@add_arg_scope
def encoder_block(x, num_channel, kernel_size, stride, is_training, scope,
                 padding='same'):
    with tf.variable_scope(scope):
        x = basic_block(x, num_channel, kernel_size, stride, is_training,
                        scope='bb1', padding=padding)
        x = basic_block(x, num_channel, kernel_size, 1, is_training,
                        scope='bb2', padding=padding)
    return x


@add_arg_scope
def upconv_bn_relu(x, num_channel, kernel_size, stride, is_training,
                   scope, padding='same', use_bias=False):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(x, num_channel, [kernel_size, kernel_size],
                                      stride, activation=None,
                                      name='conv_transpose', padding=padding,
                                      use_bias=False, kernel_initializer=he_normal())
        x = tf.layers.batch_normalization(
            x, momentum=0.9, training=is_training, name='bn'
        )
        x = tf.nn.relu(x, name='relu')
    return x


@add_arg_scope
def decoder_block(x, num_channel_m, num_channel_n, kernel_size,
                  stride=1, is_training=True, scope=None, padding='same'):
    with tf.variable_scope(scope):
        x = upconv_bn_relu(x, num_channel_m // 4, 1, 1,
                           is_training=is_training,
                           scope='conv_transpose_bn_relu1',
                           padding=padding)
        x = upconv_bn_relu(x, num_channel_m // 4, kernel_size, stride,
                           is_training=is_training,
                           scope='conv_transpose_bn_relu2',
                           padding=padding)
        x = upconv_bn_relu(x, num_channel_n, 1, 1,
                           is_training=is_training,
                           scope='conv_transpose_bn_relu3',
                           padding=padding)
    return x


@add_arg_scope
def initial_block(x, is_training=True, scope='initial_block',
                  padding='same', use_bias=False):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(x, 64, [7, 7], strides=2, activation=None,
                             name='conv1', padding=padding, use_bias=use_bias,
                             kernel_initializer=he_normal())
        x = tf.layers.batch_normalization(
            x, momentum=0.9, training=is_training, name='bn1'
        )
        x = tf.nn.relu(x, name='relu')
        x = tf.layers.max_pooling2d(x, [3, 3], strides=2, name='maxpool',
                                    padding=padding)
    return x


def linknet(inputs, num_classes, reuse=None, is_training=True,
            scope='linknet'):

    filters = [64, 128, 256, 512]
    filters_m = [64, 128, 256, 512][::-1]
    filters_n = [64, 64, 128, 256][::-1]

    with tf.variable_scope(scope, reuse=reuse):

            # Encoder
            eb0 = initial_block(inputs, is_training=is_training,
                                scope='initial_block')
            eb1 = encoder_block(eb0, filters[0], 3, 1, is_training,
                                scope='eb1', padding='same')
            ebi = eb1
            ebs = [eb1, ]
            i = 2
            for filter_i in filters[1:]:
                ebi = encoder_block(ebi, filter_i, 3, 2, is_training,
                                scope='eb'+str(i), padding='same')
                ebs.append(ebi)
                i = i + 1
            net = ebi

            # Decoder
            dbi = decoder_block(net, filters_m[0], filters_n[0], 3,
                                2, is_training=is_training, scope='db4',
                                padding='same')
            i = len(filters_m) - 1
            for filters_i in zip(filters_m[1:-1], filters_n[1:-1]):
                dbi = dbi + ebs[i-1]
                dbi = decoder_block(dbi, filters_i[0], filters_i[1], 3,
                                    2, is_training=is_training,
                                    scope='db'+str(i), padding='same')
                i = i - 1
            dbi = dbi + ebs[0]
            dbi = decoder_block(dbi, filters_m[-1], filters_n[-1], 3, 1,
                                is_training=is_training,
                                scope='db1', padding='same')
            net = dbi

            # Classification
            with tf.variable_scope('classifier', reuse=reuse):
                net = upconv_bn_relu(net, 32, 3, 2, is_training=is_training,
                                     scope='conv_transpose')
                net = conv_bn_relu(net, 32, 3, 1, is_training=is_training,
                                   scope='conv')
                # Last layer, no batch normalization or activation
                logits = tf.layers.conv2d_transpose(net, num_classes,
                                                    kernel_size=2, strides=2,
                                                    padding='same',
                                                    name='conv_transpose',
                                                    kernel_initializer=he_normal())
                if num_classes > 1:
                    prob = tf.nn.softmax(logits, name='prob')
                else:
                    prob = tf.nn.sigmoid(logits, name='prob')

                return prob, logits