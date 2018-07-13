import tensorflow as tf
import pandas as pd

def input_fn():
    inputData = pd.read_csv("./data/bear_data.csv", usecols=[0, 1])
    outputData = pd.read_csv("./data/bear_data.csv", usecols=[2, 3])

    feature = 2
    layer1 = 4
    layer2 = 5
    layer3 = 4
    classes = 2
    rate = 1.2

    x = tf.placeholder(tf.float32, [None, feature])
    y = tf.placeholder(tf.float32, [None, classes])

weight1 = tf.Variable(tf.zeros(shape=[feature, layer1]), dtype=tf.float32, name='weight1')
bais1 = tf.Variable(tf.zeros(shape=[layer1]), dtype=tf.float32, name='bais1')

weight2 = tf.Variable(tf.zeros(shape=[layer1, layer2]), dtype=tf.float32, name='weight2')
bais2 = tf.Variable(tf.zeros(shape=[layer2]), dtype=tf.float32, name='bais2')

weight3 = tf.Variable(tf.zeros(shape=[layer2, layer3]), dtype=tf.float32, name='weight3')
bais3 = tf.Variable(tf.zeros(shape=[layer3]), dtype=tf.float32, name='bais3')

weightOut = tf.Variable(tf.zeros(shape=[layer3, classes]), dtype=tf.float32, name='weightOut')
baisOut = tf.Variable(tf.zeros(shape=[classes]), dtype=tf.float32, name='baisOut')

para_list = [weight1, weight2, weight3, weightOut, bais1, bais2, bais3, baisOut]
saver = tf.train.Saver(para_list)

_layer1 = tf.sigmoid(tf.matmul(x, weight1) + bais1)
_layer2 = tf.sigmoid(tf.matmul(_layer1, weight2) + bais2)
_layer3 = tf.sigmoid(tf.matmul(_layer2, weight3) + bais3)
_classes = tf.sigmoid(tf.matmul(_layer3, weightOut) + baisOut)

y_ = tf.nn.softmax(_classes)

#---------------
cross_entropy= tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(20000):
    _, cost = sess.run([train_step, cross_entropy], feed_dict={x: inputData, y: outputData})
    if i % 1000 == 0:
        print("Step: ", i)
        print("Cost: ", cost)
        print("-----------------")
saver.save(sess, './data/trained_weight.ckpt')

correct_prediction = tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(y,1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
acc = sess.run(accuracy, feed_dict={x:inputData, y:outputData})

print("Accuracy: ", acc)

testInput = [[250, 250],
             [300, 650],
             [132, 95],
             [134, 100]]
testOut = [[1, 0],
           [1, 0],
           [0, 1],
           [0, 1]]

saver.restore(sess, './data/trained_weight.ckpt')
decision = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

result = sess.run(decision, feed_dict={x: testInput, y:testOut})
print("Result: ", result)