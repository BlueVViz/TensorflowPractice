import tensorflow as tf


# Generator for get a data
# => Take too many time
# => Solution TextlineDataset("Path") / Dataset("Path")
def gen():
    for i, j in zip(range(10, 1010), range(1000)):
        yield (i, j)


dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))\
    .shuffle(2000)\
    .batch(10)\
    .make_one_shot_iterator()\
    .get_next()


with tf.Session() as sess:
    for _ in range(10):
        _label, _data = sess.run(dataset)
        print(_label, _data)

