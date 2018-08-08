#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
# coding=utf-8
"""This module for model training."""
import os
import time

import tensorflow as tf

from config import FLAGS
from dataset import get_iterator
from utils import print_args, load_vocab, parse_model, load_model


def train():
    # Training
    tf.set_random_seed(FLAGS.random_seed)
    saver = tf.train.Saver()
    signature = model.signature_def()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        init_ops = [tf.global_variables_initializer(),
                    tf.local_variables_initializer(), tf.tables_initializer()]
        sess.run(init_ops)
        # load saved model
        if FLAGS.finetune:
            load_model(sess, FLAGS.model_dir)
            FLAGS.lr = FLAGS.lr / 10
            print("Finetune learning rate: {} devided 10 by initial learning rate".format(FLAGS.lr))

        best_acc = 0.0
        for epoch in range(FLAGS.max_epoch):
            step = 0
            if FLAGS.use_learning_decay and (epoch+1) % FLAGS.lr_decay_epoch == 0:
                FLAGS.lr *= FLAGS.lr_decay_rate
            print('\nepoch: {}\tlearning rate: {}'.format(epoch+1, FLAGS.lr))

            sess.run(
                [iterator.initializer, model.mode],
                feed_dict={data_file_placeholder: FLAGS.train_file,
                           mode: tf.estimator.ModeKeys.TRAIN})
            while True:
                try:
                    sess.run(model.update)
                    step += 1
                    # show train batch metrics
                    if step % FLAGS.show_loss_step_num == 0:
                        _, loss, accuracy = model.train(sess)
                        now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                        # time_str = datetime.datetime.now().isoformat()
                        print('{}\tepoch {:2d}\tstep\t{:3d}\ttrain loss={:.4f}\tacc={:.2f}'.format(
                            now_time, epoch + 1, step, loss, accuracy * 100))
                except tf.errors.OutOfRangeError:
                    print("\n"+"="*25+" Finish train {} epoch ".format(epoch+1)+"="*25+"\n")
                    break

            # show dev result per epoch
            acc = 0.0
            step = 0
            sess.run(
                [iterator.initializer, model.mode],
                feed_dict={data_file_placeholder: FLAGS.train_file,
                           mode: tf.estimator.ModeKeys.EVAL})
            while True:
                try:
                    loss, accuracy = model.evaluate(sess)
                    acc += accuracy
                    step += 1
                except tf.errors.OutOfRangeError:  # including last batch
                    break
            acc = acc / step
            if acc > best_acc:
                best_acc = acc
                improved_token = '*'
            else:
                improved_token = ''
            now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
            print('{}\tepoch {:2d}\tstep {:3d}\teval\tloss={:.4f}\tacc={:.2f} {}'.format(
                now_time, epoch + 1, step, loss, acc*100, improved_token))
            if FLAGS.savemodel and improved_token == '*':
                ckpt_path = os.path.join(
                    FLAGS.checkpointDir, "model.ckpt_{:.3f}".format(acc))
                model.save(sess, ckpt_path)
                print("Saved checkpoint with acc={} to {}".format(acc, ckpt_path))

            # save model
            if FLAGS.savemodel and (epoch + 1) % FLAGS.save_model_per_epochs == 0:
                model_name = "model_{}_{}".format(
                    epoch+1, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
                ckpt_path = os.path.join(FLAGS.checkpointDir, model_name)
                model.savedmodel(sess, signature, ckpt_path)
                print("Export SavedModel with acc={} to {}".format(acc, ckpt_path))

if __name__ == '__main__':
    # tf.set_min_vlog_level(1)
    # Params Preparation
    print_args(FLAGS)
    vocab_table, _, vocab_size = load_vocab(FLAGS.vocab_file)
    FLAGS.vocab_size = vocab_size

    # Model Preparation
    data_file_placeholder = tf.placeholder(tf.string, [])
    mode = tf.placeholder(tf.string, [])

    iterator = get_iterator(
        data_file_placeholder, vocab_table, FLAGS.batch_size,
        question_max_len=FLAGS.question_max_len,
        answer_max_len=FLAGS.answer_max_len,
        shuffle_buffer_size=FLAGS.num_samples)
    model = parse_model(iterator, FLAGS, mode)
    train()