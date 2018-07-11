import tensorflow as tf

feature = 8
classes = 2
alpha = 0.01

x = tf.placeholder(tf.float32, shape=[None, feature])
y = tf.placeholder(tf.float32, shape=[None, classes])

w = tf.Variable(tf.random_normal([feature, classes]), dtype=tf.float32, name='Weight')
b = tf.Variable(tf.random_normal([classes], dtype=tf.float32, name='bias'))