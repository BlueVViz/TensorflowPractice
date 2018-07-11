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

# Train line
train = tf.GradientDescentOptimizer(alpha).minimize(cost)


# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})
    print("\nHypothesis: ", h, "\nCorrect(Y: ", c, "\nAccuracy: ", a)



