import tensorflow as tf

# Variable
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]


x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variables(tf.random_normal([2, 1]), name='weight')
bias = tf.Variables(tf.random_normal([1]), name='bias')
