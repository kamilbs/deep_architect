import numpy as np
import tensorflow as tf
import darch.modules as modules


class Embedding(modules.BasicModule):

    def __init__(self, trainable, emb_dimension, param_init_fns, vocabulary_size, pretrained_array=None, expand_one_dim=False):
        super(Embedding, self).__init__()
        self.order.extend(["trainable", "emb_dimension", "param_init_fn"])
        self.domains.extend([trainable, emb_dimension ,param_init_fns])

        self.using_pretrained = False

        if pretrained_array is not None:
            self.pretrained_shape = pretrained_array.shape
            self.pretrained_array = pretrained_array
            self.using_pretrained = True

        self.expand_one_dim = expand_one_dim # If an additional dimension is needed (e.g. 1 channel for CNN)
        self.vocabulary_size = vocabulary_size

    def initialize(self, in_d, scope):
        if len(in_d) != 1:
            raise ValueError
        elif not self.using_pretrained and self.domains[0] != [True]:
            # When learning embedding from scratch trainable should be set to True only
            raise ValueError
        elif self.using_pretrained and (self.domains[1] != [self.pretrained_shape[1]] or self.vocabulary_size != self.pretrained_shape[0]):
            # When using pretrained embedding dimension should be set to the one of the pretrained only , same for vocabulary size
            raise ValueError
        else:
            super(Embedding, self).initialize(in_d, scope)

    def get_outdim(self):
        sequence_length = self.in_d[0]
        embed_dimension = self.domains[1][self.chosen[1]]
        if self.expand_one_dim:
            out_d = (sequence_length, embed_dimension, 1)
        else:
            out_d = (sequence_length, embed_dimension)
        return out_d

    def compile(self, in_x, train_feed, eval_feed):
        trainable, emb_dimension, param_init_fn = [dom[i] for (dom, i) in zip(self.domains, self.chosen)]
        W = tf.Variable(param_init_fn([self.vocabulary_size, emb_dimension]), trainable=trainable)
        if self.using_pretrained:
            W = W.assign(self.pretrained_array)
            with tf.Session() as sess:
                sess.run(tf.assign(W, self.pretrained_array))

        embedded_tokens = tf.nn.embedding_lookup(W, in_x)
        if self.expand_one_dim:
            embedded_tokens = tf.expand_dims(embedded_tokens, -1)

        return embedded_tokens









