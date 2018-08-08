#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module implements CNN encoder and RNN encoder."""
import abc

import tensorflow as tf


class Encoder(object):
    """Abstract encoder class."""
    def __init__(self, iterator, params, mode):
        """Initialize encoder
        Args:
          iterator: instance of class BatchedInput.  
          params: parameters.
          mode: train | eval | predict mode defined with tf.estimator.ModeKeys.
        """
        self.iterator = iterator
        self.params = params
        self.mode = mode

    @abc.abstractmethod
    def __call__(self, x):
        """Forward representation.
        Args:
            x: data Tensor
        """
        pass


class CNNEncoder(Encoder):
    """This class implements CNN encoder."""

    def __call__(self, x):
        params = self.params

        # Use tf high level API tf.layers
        pooled_outputs = []
        for i, filter_size in enumerate(map(int, params.filter_sizes.split(','))):
            with tf.variable_scope("conv-maxpool"):
                conv = tf.layers.conv1d(
                    x, params.num_filters, filter_size,
                    activation=tf.nn.relu,
                    bias_initializer=tf.constant_initializer(0.1),
                    name="kernel_%d" % filter_size)  # (batch_size, seq_length，num_filters)
                pool = tf.layers.max_pooling1d(
                    conv, params.max_length-filter_size+1, 1)  # (batch_size, 1， num_filters)
                pooled_outputs.append(pool)

        # Combine all the pooled features
        num_filters_total = params.num_filters * len(params.filter_sizes.split(','))
        outputs = tf.concat(pooled_outputs, 2)  # (batch_size, 1， num_filters, num_filters_total)
        self.outputs = tf.reshape(outputs, [-1, num_filters_total], name="output")  # (batch_size, num_filters_total)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.outputs = tf.nn.dropout(self.outputs, params.dropout, name="output")

        # # or use tf low level API tf.nn
        # # Expand dimension,  shape(batch_size, seq_len, dim, 1)
        # self.data_expanded = tf.expand_dims(x, -1)

        # for i, filter_size in enumerate(map(int, params.filter_sizes.split(','))):
        #     with tf.variable_scope("conv-maxpool-%s" % filter_size):
        #         # Convolution layer
        #         filter_shape = [filter_size, params.input_dim, 1, params.num_filters]
        #         W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        #         b = tf.get_variable("bias", [params.num_filters], initializer=tf.constant_initializer(0.1))
        #         conv = tf.nn.conv2d(
        #             self.data_expanded, W, strides=[1, 1, 1, 1],
        #             padding="VALID", name="conv")
        #         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # (batch_size, length, 1, num_filters)
        #         # Max-pooling layer
        #         pooled = tf.nn.max_pool(
        #             h, ksize=[1, params.max_length-filter_size+1, 1, 1],
        #             strides=[1, 1, 1, 1], padding='VALID', name="pool")  # (batch_size, 1, 1, num_filters)
        #         pooled_outputs.append(pooled)
        # # Combine all the pooled features
        # num_filters_total = params.num_filters * len(params.filter_sizes.split(','))
        # outputs = tf.concat(pooled_outputs, 3)
        # self.outputs = tf.reshape(outputs, [-1, num_filters_total])  # (batch_size, num_filters_total)
        # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #     self.outputs = tf.nn.dropout(self.outputs, params.dropout)
        # with tf.variable_scope("output"):
        #     # validate_shape=False allows initialized with a value of unknown shape.
        #     W = tf.get_variable(
        #             "W", shape=[num_filters_total, params.num_classes],
        #             initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable("bias", [params.num_classes], initializer=tf.constant_initializer(0.1))
        # self.logits = tf.nn.xw_plus_b(self.outputs, W, b, name="logits")
        return outputs


class RNNEncoder(Encoder):
    """This class implements RNN encoder."""

    def __call__(self, x):
        params = self.params
        self.length = self.iterator.sequence_length  # actual sequence length vector
        # self.hidden_size = 2 * params.hidden_units if params.bidirectional else params.hidden_units

        if not params.bidirectional:
            cell = tf.nn.rnn_cell.LSTMCell(params.num_hidden)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=params.dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * params.num_layers)
            output, _ = tf.nn.dynamic_rnn(
                cell,
                self.iterator.data,
                dtype=tf.float32,
                sequence_length=self.length)
        else:
            fw_cells = []
            bw_cells = []
            for _ in range(params.num_layers):
                fw = tf.nn.rnn_cell.LSTMCell(params.hidden_units)
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    fw = tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=params.dropout)
                fw_cells.append(fw)
                bw = tf.nn.rnn_cell.LSTMCell(params.hidden_units)
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    bw = tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=params.dropout)
                bw_cells.append(bw)
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_cells)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_cells)
        _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, sequence_length=self.length, dtype=tf.float32)
        # output_state_fw, (LSTMStateTuple(c,h) * num_layers)
        outputs = tf.concat(
            [output_state_fw[-1].h, output_state_bw[-1].h], 1, name="output")  # batch_size * hidden_units

        return outputs

