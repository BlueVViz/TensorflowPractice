import tensorflow as tf
import pandas as pd

def input_fn():
    dataSet = pd.read_csv("./data/bear_data.csv", usecols=[0, 1])
    label = pd.read_csv("./data/bear_data.csv", usecols=[2, 3])

    return

def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PRED = mode == tf.estimator.ModeKeys.PREDICT

    _layer1 = tf.layers.dense(features, units=2, activation=tf.nn.sigmoid)
    _layer2 = tf.layers.dense(_layer1, units=2, activation=tf.nn.sigmoid)
    _layer3 = tf.layers.dense(_layer2, units=2, activation=tf.nn.sigmoid)
    _classes = tf.layers.dense(_layer3, units=1)

#    y_ = tf.nn.softmax(_classes)
    loss = tf.losses.sigmoid_cross_entropy(labels, _classes)

    if TRAIN:
        gs = tf.train.get_global_step()
        train_step = tf.train.GradientDescentOptimizer(rate).minimize(loss, gs)
        return tf.estimator.EstimatorSpec(mode, loss, train_step)

    elif EVAL:
        pred = tf.nn.sigmoid(_classes)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode, loss, eval_metric_ops={"acc": accuracy})

    elif PRED:
        prob = tf.nn.sigmoid(y_)
        _class = tf.round(prob)
        return tf.estimator.EstimatorSpec(mode, prediction={"prob": prob, "class": _class})



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    est = tf.estimator.Estimator(model_fn)
    est.train(input_fn, steps=1000)
    est.evaluate(input_fn, steps=10)

    data = np.array([200, 200], np.float32)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn({"feature": data}, shuffle=False)
    for pred in est.predict(pred_input_fn):
        print("pred: {}, class: {}, ", format(pred["prob"], pred["class"]))