#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This class implement AI-NN model for CQA answer selection.
References:
    `Attentive Interactive Neural Networks for Answer Selection in Community Question Answering`, 2017
"""
import tensorflow as tf

from encoder import CNNEncoder, RNNEncoder
from model import Model


class AI_NN(Model):
    """Abstract class for AI-NN model."""

    def __init__(self, iterator, params, mode):
        self.cnn_encoder = CNNEncoder(iterator, params, mode)
        self.rnn_encoder = CNNEncoder(iterator, params, mode)
        super(AI_NN, self).__init__(iterator, params, mode)


class AI_CNN(AI_NN):

    def _build_logits(self):
        self.cnn_out = self.cnn_encoder(self.data)



