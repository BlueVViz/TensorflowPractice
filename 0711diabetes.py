import tensorflow as tf
import numpy as np
import pandas as pd

x_data = pd.read_csv("./data/data-03-diabetes.csv", usecols=[0, 1,2,3,4,5,6, 7])
y_data = pd.read_csv("./data/data-03-diabetes.csv", usecols=[8, 9])

# xy = np.loadtxt("./data/data-03-diabetes.csv", delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, -1]

feature = 8
hidden1 = 16
hidden2 = 6
classes = 2
alpha = 55

x = tf.placeholder(tf.float32, shape=[None, feature])
y = tf.placeholder(tf.float32, shape=[None, classes])

w1 = tf.Variable(tf.random_normal([feature, hidden1]), dtype=tf.float32, name='weight1')
b1 = tf.Variable(tf.random_normal([hidden1], dtype=tf.float32, name='bias1'))

w2 = tf.Variable(tf.random_normal([hidden1, hidden2]), dtype=tf.float32, name='weight2')
b2 = tf.Variable(tf.random_normal([hidden2]), dtype=tf.float32, name='bias2')

w3 = tf.Variable(tf.random_normal([hidden2, classes]), dtype=tf.float32, name='weight3')
b3 = tf.Variable(tf.random_normal([classes]), dtype=tf.float32, name='bias3')

layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)
hypo = tf.sigmoid(tf.matmul(layer2, w3) + b3)

hypothesis = tf.nn.softmax(hypo)

# cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

# predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
predict = tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1)), tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
accuracy = tf.reduce_mean(predict)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}))

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis: ", hypothesis, "\nPredict: ", c, "Accurary: ", a)


