# Learning Tensorflow - data type
# Variable
import tensorflow as tf

# Set data => Going to create Graph
val1 = tf.Variable(1)
val2 = tf.Variable(2)
val3 = tf.Variable(3)
val4 = val1 * val2 + val3

# Create Session
sess = tf.Session()

# Initialize all variable
init = tf.global_variables_initializer()
sess.run(init)

# Run / Get a result
result = sess.run(val4)
print(result)

# Finish using device - Session close
sess.close()