#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module contains efficient data read and transform using tf.data API."""
import collections

import tensorflow as tf


class BatchedInput(
    collections.namedtuple(
        "BatchedInput", ("initializer", "question", "answer", "target", "question_length", "answer_length"))):
    pass


def get_infer_iterator(data_file, vocab_table, batch_size, question_max_len=None, answer_max_len=None, padding=False):
    """Iterator for inference.
    Args:
        data_file: data file, each line contains question, answer
        vocab_table: tf look-up table
        question_max_len: question max length
        answer_max_len: answer max length
        padding: Bool
            set True for cnn or attention based model to pad all samples into same length, must set seq_max_len
            set False for rnn model 
    Returns:
        BatchedInput instance
            question ids, answer_ids, question length, answer length.
    """
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(
        lambda q, a: (tf.string_split([q]).values, tf.string_split([a]).values))
    if question_max_len:
        dataset = dataset.map(lambda q, a: (q[:question_max_len], a))
    if answer_max_len:
        dataset = dataset.map(lambda q, a: (q, a[:question_max_len]))
    # Convert the word strings to ids
    dataset = dataset.map(
        lambda q, a: (tf.cast(vocab_table.lookup(q), tf.int32),
                      tf.cast(vocab_table.lookup(a), tf.int32),
                      tf.size(q),
                      tf.size(a)))

    question_pad_size = question_max_len if padding else None
    answer_pad_size = answer_max_len if padding else None
    batched_dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([question_pad_size]),
            tf.TensorShape([answer_pad_size]),
            tf.TensorShape([]), tf.TensorShape([])),
        padding_values=(0, 0, 0, 0))

    batched_iter = batched_dataset.make_initializable_iterator()
    q_ids, a_ids, q_len, a_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        question=q_ids, answer=a_ids, target=None, question_length=q_len, answer_length=a_len)


def get_iterator(data_file,
                 vocab_table,
                 batch_size,
                 random_seed=123,
                 num_buckets=1,
                 question_max_len=None,
                 answer_max_len=None,
                 padding=False,
                 num_parallel_calls=4,
                 shuffle_buffer_size=None,
                 reshuffle_each_iteration=True):
    """Iterator for train and eval.
    Args:
        data_file: data file, each line contains question, answer, label
        vocab_table: tf look-up table
        question_max_len: question max length
        answer_max_len: answer max length
        padding: Bool
            set True for cnn or attention based model to pad all samples into same length, must set seq_max_len
            set False for rnn model 
        num_buckets: bucket according to sequence length
        shuffle_buffer_size: buffer size for shuffle
    Returns:
        BatchedInput instance
            question ids, answer_ids, target, question length, answer length.
    """
    shuffle_buffer_size = shuffle_buffer_size or batch_size * 1000
    output_buffer_size = batch_size * 100

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.shuffle(shuffle_buffer_size, random_seed, reshuffle_each_iteration)
    dataset = dataset.map(
        lambda q, a, l: (tf.string_split([q]).values, tf.string_split([q]).values, l),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if question_max_len:
        dataset = dataset.map(
            lambda q, a, l: (q[:question_max_len], a, l),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if answer_max_len:
        dataset = dataset.map(
            lambda q, a, l: (q, a[:answer_max_len], l),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(
        lambda q, a, l: (tf.cast(vocab_table.lookup(q), tf.int32),
                         tf.cast(vocab_table.lookup(a), tf.int32),
                         tf.cast(l, tf.int32),
                         tf.size(q),
                         tf.size(a)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    question_pad_size = question_max_len if padding else None
    answer_pad_size = answer_max_len if padding else None

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([question_pad_size]),
                tf.TensorShape([answer_pad_size]),
                tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
            padding_values=(0, 0, 0, 0, 0))

    # Bucket by sequence length (buckets for lengths 0-9, 10-19, ...)
    # or use tf.contrib.data.bucket_by_sequence_length
    if num_buckets > 1:

        def key_func(unused_1, unused_2, unused_3, q_len, a_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if question_max_len:
                bucket_width = (question_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(q_len // bucket_width, a_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    else:
        batched_dataset = batching_func(dataset)

    batched_iter = batched_dataset.make_initializable_iterator()  # sess.run(iterator.initializer)
    q_ids, a_ids, target, q_len, a_len = batched_iter.get_next()

    return BatchedInput(
        initializer=batched_iter.initializer,
        question=q_ids, answer=a_ids, target=target, question_length=q_len, answer_length=a_len)

