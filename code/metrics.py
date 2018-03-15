import tensorflow as tf


def dice_coef(y_true, y_pred, axis=[1, 2, 3], smooth=1e-5):
    # Tensorflow calculate dice_coefficient
    with tf.name_scope('dice_coef'):
        '''
        y_true = tf.round(y_true)
        y_pred = tf.round(y_pred)
        inse = tf.reduce_sum(y_true * y_pred, axis)
        l = tf.reduce_sum(y_true, axis=axis)
        r = tf.reduce_sum(y_pred, axis=axis)
        dice = (2 * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice
        '''
        y_true = tf.round(tf.reshape((y_true), [-1]))
        y_pred = tf.round(tf.reshape((y_pred), [-1]))

        isct = tf.reduce_sum(y_true * y_pred)

        return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))