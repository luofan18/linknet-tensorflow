import tensorflow as tf


def random_flip_left_right(image, label, seed=None):
    '''
    Randomly flip an image and label horizontally
    :param image:
    :param label:
    :param seed:
    :return:
    '''
    with tf.name_scope('random_flip_left_right') as scope:
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)
        random_number = tf.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.less(random_number, 0.5)
        image = tf.cond(
            mirror_cond,
            lambda: tf.image.flip_left_right(image),
            lambda: image,
            name='flip_image_or_not'
        )
        label = tf.cond(
            mirror_cond,
            lambda: tf.image.flip_left_right(label),
            lambda: label,
            name='flip_label_or_not'
        )
        return image, label