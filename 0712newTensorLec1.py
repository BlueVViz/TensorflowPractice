import tensorflow as tf
import numpy as np

numFeature = 10
numLAbel = 1
alpha = 0.01


def input_fn():

    dataset = tf.data.TextLineDataset("")\
        .batch(20)\
        .repeat(999999)\
        .make_one_shot_iterator()\
        .get_next()

    lines = tf.decode_csv(dataset, record_defaults=[[0]*(numFeature + numLAbel)])

    feature = tf.stack(lines[1:], axis=1)
    label = tf.expand_dims(lines[0], axis=-1)

    feature = tf.cast(feature, tf.float32)
    label = tf.cast(label, tf.float32)
    return {"feature": feature}, label


def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PRED = mode == tf.estimator.ModeKeys.PREDICT

    # Write a Model, hidden layers
    layer1 = tf.layers.dense(features, units=10, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, units=10, activation=tf.nn.relu)
    layer3 = tf.layers.dense(layer2, units=10, activation=tf.nn.relu)
    out = tf.layers.dense(layer3, units=1)

    loss = tf.losses.sigmoid_cross_entropy(labels, out)

    # For Train
    if TRAIN:
        gs = tf.train.get_global_step()
        train = tf.train.GradientDescentOptimizer(alpha).minimize(loss, gs)
        return tf.estimator.EstimatorSpec(mode, loss, train)

    # For Evaluation
    elif EVAL:
        pred = tf.nn.sigmoid(out)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        return tf.estimator.EstimatorSpec(mode, loss, eval_metric_ops={"acc": accuracy})

    # For Prediction
    elif PRED:
        prob = tf.nn.sigmoid(out)
        _class = tf.round(prob)

        return tf.estimator.EstimatorSpec(mode, prediction={"prob": prob, "class": _class})

if __name__== "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    est = tf.estimator.Estimator(model_fn)
    est.train(input_fn, steps=1000)
    est.evaluate(input_fn, steps=10)

    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.float32)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn({"feature": data}, shuffle=False)
    for pred in est.predict(pred_input_fn):
        print("pred: {}, class: {}, ", format(pred["prob"], pred["class"]))