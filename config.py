#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
import tensorflow as tf

# data params
tf.flags.DEFINE_string("train_file", None, "train file")
tf.flags.DEFINE_string("test_file", None, "test file")
tf.flags.DEFINE_string("predict_file", None, "inference file")
tf.flags.DEFINE_string("vocab_file", None, "vocab file")
tf.flags.DEFINE_string("embed_file", None, "embed file")
tf.flags.DEFINE_string("output_file", None, "output file")
tf.flags.DEFINE_integer("question_max_len", 40, "max question length [40]")
tf.flags.DEFINE_integer("answer_max_len", 40, "max answer length [40]")
tf.flags.DEFINE_integer("num_samples", 10000, "num of samples [10000]")
tf.flags.DEFINE_integer("embedding_dim", 128, "embedding dim [128]")

# model params
tf.flags.DEFINE_string("model_dir", "model", "model path")
tf.flags.DEFINE_integer("model_type", 4, "model type, model index [2]")
# common
tf.flags.DEFINE_float("dropout", 0.8, "dropout keep prob [0.8]")
# cnn
tf.flags.DEFINE_integer("num_filters", 100, "num of conv filters")
tf.flags.DEFINE_string("filter_sizes", '2,3,4', "filter sizes")
# rnn
tf.flags.DEFINE_boolean("bidirectional", True, "Whether to use bidirectional rnn [1]")
tf.flags.DEFINE_integer("num_layers", 2, "num of hidden layers [2]")
tf.flags.DEFINE_integer("hidden_units", 256, "num of hidden units [256]")

# training params
tf.flags.DEFINE_integer("batch_size", 1000, "train batch size [1000]")
tf.flags.DEFINE_integer("max_epoch", 10, "max epoch [10]")
tf.flags.DEFINE_float("lr", 0.002, "init learning rate [0.002]")
tf.flags.DEFINE_integer("lr_decay_epoch", 3, "learning rate decay interval [3]")
tf.flags.DEFINE_float("lr_decay_rate", 0.5, "learning rate decay rate [0.5]")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer, `adam` | `rmsprop` | `sgd` [adam]")
tf.flags.DEFINE_integer("stats_per_steps", 1000, "show train info steps [1000]")
tf.flags.DEFINE_boolean("finetune", False, "finetune or not (keep train) [False]")
tf.flags.DEFINE_integer("save_model_per_epochs", 1, "every epochs to save model [1]")
tf.flags.DEFINE_boolean("use_learning_decay", True, "use learning decay or not [True]")
tf.flags.DEFINE_boolean("use_grad_clip", True, "whether to clip grads [False]")
tf.flags.DEFINE_integer("grad_clip_norm", 5, "max grad norm if use grad clip [5]")
tf.flags.DEFINE_integer("num_keep_ckpts", 5, "max num ckpts [5]")
tf.flags.DEFINE_integer("random_seed", 123, "random seed [123]")

# predict params
tf.flags.DEFINE_integer("predict_batch_size", 1000, "test batch size [1000]")

# auto params, do not need to set
tf.flags.DEFINE_integer("vocab_size", None, "vocabulary size")


FLAGS = tf.flags.FLAGS
