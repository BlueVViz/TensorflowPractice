import tensorflow as tf
import pandas as pd


inputData = pd.read_csv("./data/bear_data.csv", usecols=[0, 1])
outputData = pd.read_csv("./data/bear_data.csv", usecols=[2, 3])

feature = 2
layer1 = 4
layer2 = 5
layer3 = 4
output = 2
rate = 1.2

x = tf.placeholder(tf.float32, [None, feature])
y = tf.placeholder(tf.float32, [None, output])

weight1 = tf.Variable(tf.zeros([feature, layer1]), tf.float32, 'weight1')
bais1 = tf.Variable(tf.zeros([layer1]), tf.float32, 'bais1')

weight2 = tf.Variable(tf.zeros([layer1, layer2]), tf.float32, 'weight2')
bais2 = tf.Variable(tf.zeros([layer2]), tf.float32, 'bais2')

weight3 = tf.Variable(tf.zeros([layer2, layer3]), tf.float32, 'weight3')
bais3 = tf.Variable(tf.zeros([layer3]), tf.float32, 'bais3')

weightOut = tf.Variable(tf.zeros([layer3, output]), tf.float32, 'weightOut')
baisOut = tf.Variable(tf.zeros([output]), tf.float32, 'baisOut')

para_list = [weight1, weight2, weight3, weightOut, bais1, bais2, bais3, baisOut]
saver = tf.train.Saver(para_list)

_layer1 = tf.sigmoid(tf.matmul(x, weight1) + bais1)
_layer2 = tf.sigmoid(tf.matmul(_layer1, weight2) + bais2)
_layer3 = tf.sigmoid(tf.matmul(_layer2, weight3) + bais3)
_output = tf.sigmoid(tf.matmul(_layer3, weightOut) + baisOut)

y = tf.nn.softmax(_output)





