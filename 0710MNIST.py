import tensorflow as tf


# Model line

imageSize = 784
number = 10
rate = 0.5

mnist = tf.data.Dataset("./MNIST_data/")
x = tf.placeholder(tf.float32, [None, imageSize])
w = tf.Variable(tf.zeros(shape=[imageSize, number]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros(shape=[number]), dtype=tf.float32, name='b')
y = tf.sigmoid(tf.matmul(x, w) + b)
_y = tf.placeholder(tf.float32, [None, number])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(_y * tf.log(y)), reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Training line
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})



