#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem converted to TensorFlow Estimator
Uses a distribution strategy according to the recipe here:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute
"""

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import p1b3
from tensorflow.python.client import device_lib
from pudb import set_trace

# Model and Training parameters
SEED = 2016
BATCH_SIZE = 100
EPOCHS = 20
WORKERS = 1
OUT_DIR = '.'
LEARNING_RATE = 0.01
D1, D2, D3, D4 = 6000, 500, 100, 50  # Hidden units per layer
# Total parameters = (Input_dim * D1) + (D1 * D2) +
#                    (D2 * D3) + (D3 * D4) + (D4 * Output_dim) +
#                    Output_dim
#                    29532 *  6000 + 6000 * 500 +
#                    500 * 100 + 100 * 50 + 50 *  1 +
#                    1

# Type of feature scaling (options: 'maxabs': to [-1,1]
#                                   'minmax': to [0,1]
#                                   None    : standard normalization
SCALING = 'std'
# Features to (randomly) sample from cell lines or drug descriptors
# FEATURE_SUBSAMPLE = 500
FEATURE_SUBSAMPLE = 0

MIN_LOGCONC = -5.
MAX_LOGCONC = -4.
CATEGORY_CUTOFFS = [0.]
VAL_SPLIT = 0.2
TEST_CELL_SPLIT = 0.15


np.set_printoptions(threshold=np.nan)
np.random.seed(SEED)


def calc_param_count():
    [print(layer.name, layer.shape) for layer in tf.trainable_variables()]
    return np.sum([np.prod(layer.shape) for layer in tf.trainable_variables()])


def input_fn(data_getter):
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: data_getter,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(
            [BATCH_SIZE, 29532]), tf.TensorShape([BATCH_SIZE, ])),
    ).repeat()
    return dataset


def fc_model_fn(features, labels, mode):
    """Model function for a fully-connected network"""
    input_layer = tf.reshape(features, [-1, 29532])
    dense_1 = tf.layers.dense(inputs=input_layer, units=D1, name='dense_1',
                              use_bias=False, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(inputs=dense_1, units=D2, name='dense_2',
                              use_bias=False, activation=tf.nn.relu)
    dense_3 = tf.layers.dense(inputs=dense_2, units=D3, name='dense_3',
                              use_bias=False, activation=tf.nn.relu)
    dense_4 = tf.layers.dense(inputs=dense_3, units=D4, name='dense_4',
                              use_bias=False, activation=tf.nn.relu)

    regressed_val = tf.layers.dense(inputs=dense_4, units=1, name='dense_5',
                                    use_bias=False)

    tf.logging.info('Total Param Count: {}'.format(calc_param_count()))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "output": regressed_val,
    }

    tf.logging.info('mode: {}'.format(mode))
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions['output'])

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(
        labels, tf.reshape(regressed_val, [BATCH_SIZE]))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    mse = tf.metrics.mean_squared_error(
        labels=labels, predictions=predictions['output'])
    eval_metric_ops = {'mse': mse}
    tf.summary.scalar('mse', mse)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():

    print('Available GPUs', get_available_gpus())

    tf.logging.set_verbosity(tf.logging.DEBUG)
    loader = p1b3.DataLoader(val_split=VAL_SPLIT,
                             test_cell_split=TEST_CELL_SPLIT,
                             cell_features=['expression'],
                             drug_features=['descriptors'],
                             feature_subsample=FEATURE_SUBSAMPLE,
                             scaling=SCALING,
                             scramble=False,
                             min_logconc=MIN_LOGCONC,
                             max_logconc=MAX_LOGCONC,
                             subsample='naive_balancing',
                             category_cutoffs=CATEGORY_CUTOFFS)

    tf.logging.info('Loader input dim: {}'.format(loader.input_dim))
    gen_shape = None

    train_gen = p1b3.DataGenerator(loader, batch_size=BATCH_SIZE,
                                   shape=gen_shape, name='train_gen').flow()
    val_gen = p1b3.DataGenerator(loader, partition='val',
                                 batch_size=BATCH_SIZE,
                                 shape=gen_shape, name='val_gen').flow()
    val_gen2 = p1b3.DataGenerator(loader, partition='val',
                                  batch_size=BATCH_SIZE,
                                  shape=gen_shape, name='val_gen2').flow()
    test_gen = p1b3.DataGenerator(loader, partition='test',
                                  batch_size=BATCH_SIZE,
                                  shape=gen_shape, name='test_gen').flow()

    # Prep for distribution using mirrorred strategy
    devices = ["/device:GPU:0", "/device:GPU:1",
               "/device:GPU:2", "/device:GPU:3"]
    distribution = tf.contrib.distribute.MirroredStrategy(
        devices)  # alternately specify num_gpus
    config = tf.estimator.RunConfig(train_distribute=distribution)

    # Create the Estimator
    p1b3_regressor = tf.estimator.Estimator(
        model_fn=fc_model_fn,
        model_dir="/tmp/fc_regression_model",
        config=config)

    # Train & eval
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_gen))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(val_gen))
    tf.estimator.train_and_evaluate(p1b3_regressor, train_spec, eval_spec)


if __name__ == '__main__':
    main()
