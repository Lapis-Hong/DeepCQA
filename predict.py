#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module for model prediction."""

import time
import codecs

import tensorflow as tf

from config import FLAGS
from dataset import get_infer_iterator
from utils import print_args, load_vocab, parse_model, load_model


def predict():
    writer = codecs.getwriter("utf-8")(tf.gfile.GFile(FLAGS.output_file, "wb"))
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # load model
        load_model(sess, FLAGS.model_dir)

        print('Start Predicting...')
        step = 0
        sess.run(iterator.initializer)
        while True:
            try:
                item_id, label = model.predict(sess)
                writer.write(indices=[0, 1], values=[item_id, label])

            except tf.errors.OutOfRangeError:
                break
            step += 1
            if step % 10 == 0:
                now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                print('{} predict {:2d} lines'.format(now_time, step*FLAGS.predict_batch_size))
        print("Done. Write output into {}".format(FLAGS.output_file))
        writer.close()

if __name__ == '__main__':
    # Params Preparation
    print_args(FLAGS)
    vocab_table, _, vocab_size = load_vocab(FLAGS.vocab_file)
    FLAGS.vocab_size = vocab_size

    # Model Preparation
    iterator = get_infer_iterator(
        FLAGS.predict_file, vocab_table, FLAGS.predict_batch_size,
        question_max_len=FLAGS.question_max_len,
        answer_max_len=FLAGS.answer_max_len,
    )
    mode = tf.estimator.ModeKeys.PREDICT
    model = parse_model(iterator, FLAGS, mode)
    predict()
