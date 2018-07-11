import tensorflow as tf

# Variable
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
alpha = 0.1

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.random_normal([2, 1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, weight) + bias)
cost = -tf.reduce_mean(y * tf.log(hypothesis)) + (1 - y) * tf.log(1 - hypothesis)

