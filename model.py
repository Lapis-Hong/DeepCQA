#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module implements the abstract class Model."""
import abc

import tensorflow as tf

from model_utils import create_or_load_embed


class Model(object):
    """CQA abstract base class."""

    def __init__(self, iterator, params, mode):
        """Initialize model, build graph.
        Args:
          iterator: instance of class BatchedInput.  
          params: parameters.
          mode: train | eval | predict mode defined with tf.estimator.ModeKeys.
        """
        self.iterator = iterator
        self.params = params
        self.mode = mode
        self.scope = self.__class__.__name__  # instance class name
        self.initializer = None  # subclass may change
        self.regularizer = None

        with tf.variable_scope(
                self.scope,
                initializer=self.initializer,
                regularizer=self.regularizer,
                reuse=tf.AUTO_REUSE):
            embeddings = create_or_load_embed(
                params.vocab_file, params.embed_file, params.vocab_size, params.input_dim)
            self.data = tf.nn.embedding_lookup(embeddings, iterator.data)  # [batch_size, seq_length, embedding_size]
            # # Create binary mask [batch_size, length]
            # mask = tf.to_float(tf.not_equal(iterator.data, 0))
            # self.data *= tf.expand_dims(mask, -1)  # mask embedding

            self.logits = self._build_logits()  # (batch_size, num_classes)
            self.scores = tf.nn.softmax(self.logits, name="score")
            self.pred = tf.argmax(self.scores, 1, output_type=tf.int32, name="predict")  # batch_size

            if mode != tf.estimator.ModeKeys.PREDICT:  # self.update, self.loss, self.scores, self.accuracy
                with tf.name_scope("accuracy"):
                    correct_prediction = tf.equal(iterator.target, self.pred)
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="acc")
                with tf.name_scope("loss"):
                    # regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    self.target = tf.one_hot(iterator.target, params.num_classes, dtype=tf.float32)
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.target, logits=self.logits)) + tf.losses.get_regularization_loss()
                    # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    #     labels=iterator.target, logits=self.logits)) + tf.losses.get_regularization_loss()

                if params.optimizer == "rmsprop":
                    opt = tf.train.RMSPropOptimizer(params.lr)
                elif params.optimizer == "adam":
                    opt = tf.train.AdamOptimizer(params.lr)
                elif params.optimizer == "sgd":
                    opt = tf.train.MomentumOptimizer(params.lr, 0.9)
                else:
                    raise ValueError("Unsupported optimizer %s" % params.optimizer)

                train_vars = tf.trainable_variables()
                gradients = tf.gradients(self.loss, train_vars)
                # gradients, _ = opt.compute_gradients(self.loss, train_vars)
                if params.use_grad_clip:
                    gradients, grad_norm = tf.clip_by_global_norm(
                        gradients, params.grad_clip_norm)

                self.global_step = tf.Variable(0, trainable=False)
                self.update = opt.apply_gradients(
                    zip(gradients, train_vars), global_step=self.global_step)

        self.model_stats()  # print model statistics info

    @abc.abstractmethod
    def _build_logits(self):
        """Subclass must implement this method, 
        Returns: 
            A logits Tensor, shape: [batch_size, num_classes].
        """
        pass

    def train(self, sess):
        return sess.run([self.update, self.loss, self.accuracy])

    def evaluate(self, sess):
        return sess.run([self.loss, self.accuracy])

    def predict(self, sess):
        return sess.run([self.iterator.id, self.pred])

    @staticmethod
    def model_stats():
        """Print trainable variables and total model size."""
        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=self.params.num_keep_ckpts)
        saver.save(sess, path, global_step=self.global_step.eval())

    def signature_def(self):  # TODO: how to write signature def properly
        tensor_info_x = tf.saved_model.utils.build_tensor_info(self.data)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.logits)
        return tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'data': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    def savedmodel(self, sess, signature, path):
        # export_dir = os.path.join(path, str(self.params.model_version))
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict': signature},
            clear_devices=True)
        builder.save()
