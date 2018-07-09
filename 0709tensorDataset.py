import tensorflow as tf


def gen():
    for i in range(10):
        yield i


dataset = tf.data.Dataset.from_generator(gen, tf.float32)\
    .make_one_shot_iterator()\
    .get_next()


with tf.Session() as sess:
    _data = sess.run(dataset)
    print(_data)

