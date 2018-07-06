# Learning Tensorflow - data type
# PlaceHolder
import tensorflow as tf

# Variable create
val1 = 4
val2 = 3
val3 = 2

ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)
ph3 = tf.placeholder(tf.float32)

result = ph1 * ph2 + ph3

# Match placeholder to value
dic = {ph1: val1, ph2: val2, ph3: val3}

# Start Session
sess = tf.Session()
result = sess.run(result, dic)

print(result)