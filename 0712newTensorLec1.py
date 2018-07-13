import tensorflow as tf

feature
labels
alpha = 0.01


def input_fn():




def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL

    # Write a Model, hidden layers
    layer1 = tf.nn.sigmoid(tf.matmul(features, w1) + b1)
    

    # For Train
    if TRAIN:
        gs = tf.train.get_global_step()
        train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
        return tf.estimator.EstimatorSpec(mode, cost, train)

    # For
    elif EVAL:
        pred = tf.nn.sigmoid()
        accuracy =



if __name__== "__main__":
    est = tf.estimator.Estimator(model_fn)
    est.train(input_fn, steps=1000)
    est.evaluate(input_fn, steps=10)