import tensorflow as tf


def gen():
    for i, j in zip(range(10, 20), range(10)):
        yield (i, j)


dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))\
    .make_one_shot_iterator()\
    .get_next()


with tf.Session() as sess:
    for _ in range(10):
        _label, _data = sess.run(dataset)
        print(_label, _data)

