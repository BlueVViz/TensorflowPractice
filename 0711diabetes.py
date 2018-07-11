import tensorflow as tf

feature = 8
classes = 2
alpha = 0.01

x = tf.placeholder(tf.float32, shape=[None, feature])
y = tf.placeholder(tf.float32, shape=[None, classes])


