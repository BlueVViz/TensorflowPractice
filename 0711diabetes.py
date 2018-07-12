import tensorflow as tf

feature = 8
classes = 2
alpha = 0.01

x = tf.placeholder(tf.float32, shape=[None, feature])
y = tf.placeholder(tf.float32, shape=[None, classes])

w = tf.Variable(tf.random_normal([feature, classes]), dtype=tf.float32, name='Weight')
b = tf.Variable(tf.random_normal([classes], dtype=tf.float32, name='bias'))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variable_initializer())

    