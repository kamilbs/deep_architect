import tensorflow as tf
import darch.modules as modules
import warnings
import numpy as np




class Embedding(modules.BasicModule):

    def __init__(self, trainable, emb_dimension, param_init_fns, vocabulary_size, pretrained_array=None, expand_one_dim=False):
        super(Embedding, self).__init__()
        self.order.extend(["trainable", "emb_dimension", "param_init_fn"])
        self.domains.extend([trainable, emb_dimension, param_init_fns])

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
        with tf.device('/cpu:0'):
            W = tf.Variable(param_init_fn([self.vocabulary_size, emb_dimension]), trainable=trainable)
            if self.using_pretrained:
                W = W.assign(self.pretrained_array)
                with tf.Session() as sess:
                    sess.run(tf.assign(W, self.pretrained_array))

            embedded_tokens = tf.nn.embedding_lookup(W, in_x)
            if self.expand_one_dim:
                embedded_tokens = tf.expand_dims(embedded_tokens, -1)
            return embedded_tokens


class EmbeddingDropout(modules.BasicModule):

    def __init__(self, ps):
        super(EmbeddingDropout, self).__init__()
        self.order.append("keep_prob")
        self.domains.append(ps)

    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        p_name = self.namespace_id + '_' + self.order[0]
        p_var = tf.placeholder(tf.float32, name=p_name)

        # during training the value of the dropout probability (keep_prob) is
        # set to the actual chosen value.
        # during evalution, it is set to 1.0.
        p_val = self.domains[0][self.chosen[0]]
        train_feed[p_var] = p_val
        eval_feed[p_var] = 1.0
        if len(self.in_d) == 2:
            out_y = tf.nn.dropout(in_x, p_var, noise_shape=[self.in_d[0], 1])
        elif len(self.in_d) == 3:
            out_y = tf.nn.dropout(in_x, p_var, noise_shape=[self.in_d[0], 1, 1])

        return out_y

def length(sequence):
    """
    Assume 0 padding
    :param sequence: placeholder : batch_size x max_length x features
    :return:
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

class RNN(modules.BasicModule):

    def __init__(self, num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                 only_last_output=False):
        super(RNN, self).__init__()

        self.order.extend(["num_units", "input_keep_prob", "output_keep_prob", "state_keep_prob"])
        self.domains.extend([num_units, input_keep_prob, output_keep_prob, state_keep_prob])

        self.only_last_output = only_last_output

    # Additional error checking on dimension
    def initialize(self, in_d, scope):
        if len(in_d) == 3 and in_d[1] == 1:
            warnings.warn('Please check that is indeed a CNN output as input (--getting rid of one dimension--)')
            super(RNN, self).initialize(in_d, scope)
        elif len(in_d) != 2:
            raise ValueError
        else:
            super(RNN, self).initialize(in_d, scope)

    def get_cell(self, num_hidden):
        raise NotImplementedError

    def get_outdim(self):
        num_units_chosen = self.domains[0][self.chosen[0]]
        if self.only_last_output:
            return (num_units_chosen,)
        else:
            timesteps = self.in_d[0]
            return timesteps, num_units_chosen

    def compile(self, in_x, train_feed, eval_feed):
        timesteps = self.in_d[0]
        num_units_chosen, input_keep_prob_chosen,\
            output_keep_prob_chosen, state_keep_prob_chosen = [dom[i] for (dom, i) in zip(self.domains, self.chosen)]

        keep_probs_chosen = [input_keep_prob_chosen, output_keep_prob_chosen, state_keep_prob_chosen]

        # placeholder names
        p_names = [self.namespace_id + '_' + name for name in self.order[1:]]

        placeholder_list = [tf.placeholder(tf.float32, name=n) for n in p_names]

        for placeholder, keep_prob_val in zip(placeholder_list, keep_probs_chosen):
            train_feed[placeholder] = keep_prob_val
            eval_feed[placeholder] = 1

        if len(self.in_d) == 3:
            in_x = tf.squeeze(in_x, axis=[2])

        unstacked_x = tf.unstack(in_x, timesteps, 1)

        cell = tf.contrib.rnn.DropoutWrapper(self.get_cell(num_units_chosen),
                                             input_keep_prob=placeholder_list[0],
                                             output_keep_prob=placeholder_list[1],
                                             state_keep_prob=placeholder_list[2])

        outputs, _ = tf.nn.dynamic_rnn(cell, unstacked_x, sequence_length=length(unstacked_x), dtype=tf.float32)
        if self.only_last_output:
            return outputs[-1]
        else:
            return tf.stack(outputs, 1)


class BiRNN(modules.BasicModule):

    def __init__(self, num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                 only_last_output=False):
        super(BiRNN, self).__init__()

        self.order.extend(["num_units", "input_keep_prob", "output_keep_prob", "state_keep_prob"])
        self.domains.extend([num_units, input_keep_prob, output_keep_prob, state_keep_prob])

        self.only_last_output = only_last_output

    # Additional error checking on dimension
    def initialize(self, in_d, scope):
        if len(in_d) == 3 and in_d[1] == 1:
            warnings.warn('Please check that is indeed a CNN output as input (--getting rid of one dimension--)')
            super(BiRNN, self).initialize(in_d, scope)
        elif len(in_d) != 2:
            raise ValueError
        else:
            super(BiRNN, self).initialize(in_d, scope)

    def get_cell(self, num_hidden):
        raise NotImplementedError

    def get_outdim(self):
        num_units_chosen = self.domains[0][self.chosen[0]]
        if self.only_last_output:
            return (num_units_chosen * 2,)
        else:
            timesteps = self.in_d[0]
            return timesteps, num_units_chosen * 2

    def compile(self, in_x, train_feed, eval_feed):
        timesteps = self.in_d[0]
        num_units_chosen, input_keep_prob_chosen,\
            output_keep_prob_chosen, state_keep_prob_chosen = [dom[i] for (dom, i) in zip(self.domains, self.chosen)]

        keep_probs_chosen = [input_keep_prob_chosen, output_keep_prob_chosen, state_keep_prob_chosen]

        # placeholder names
        p_names = [self.namespace_id + '_' + name for name in self.order[1:]]

        placeholder_list = [tf.placeholder(tf.float32, name=n) for n in p_names]

        for placeholder, keep_prob_val in zip(placeholder_list, keep_probs_chosen):
            train_feed[placeholder] = keep_prob_val
            eval_feed[placeholder] = 1

        if len(self.in_d) == 3:
            in_x = tf.squeeze(in_x, axis=[2])

        unstacked_x = tf.unstack(in_x, timesteps, 1)

        forward_cell = tf.contrib.rnn.DropoutWrapper(self.get_cell(num_units_chosen),
                                                     input_keep_prob=placeholder_list[0],
                                                     output_keep_prob=placeholder_list[1],
                                                     state_keep_prob=placeholder_list[2])

        backward_cell = tf.contrib.rnn.DropoutWrapper(self.get_cell(num_units_chosen),
                                                      input_keep_prob=placeholder_list[0],
                                                      output_keep_prob=placeholder_list[1],
                                                      state_keep_prob=placeholder_list[2])

        # forward and backward outputs are concatenated
        outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                        unstacked_x, sequence_length=length(unstacked_x),
                                                        dtype=tf.float32)
        if self.only_last_output:
            return outputs[-1]
        else:
            return tf.stack(outputs, 1)


class LSTM(RNN):
    def __init__(self, num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                 only_last_output=False):
        super(LSTM, self).__init__(num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                                   only_last_output=only_last_output)

    def get_cell(self, num_hidden):
        return tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden)


class GRU(RNN):
    def __init__(self, num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                 only_last_output=False):
        super(GRU, self).__init__(num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                                  only_last_output=only_last_output)

    def get_cell(self, num_hidden):
        return tf.contrib.rnn.GRUCell(num_units=num_hidden)


class BiLSTM(BiRNN):
    def __init__(self, num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                 only_last_output=False):
        super(BiLSTM, self).__init__(num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                                     only_last_output=only_last_output)

    def get_cell(self, num_hidden):
        return tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden)


class BiGRU(BiRNN):
    def __init__(self, num_units,input_keep_prob, output_keep_prob, state_keep_prob,
                 only_last_output=False):
        super(BiGRU, self).__init__(num_units, input_keep_prob, output_keep_prob, state_keep_prob,
                                    only_last_output=only_last_output)

    def get_cell(self, num_hidden):
        return tf.contrib.rnn.GRUCell(num_units=num_hidden)


class ClassificationAttention(modules.BasicModule):
    def __init__(self, param_init_fns):
        super(ClassificationAttention, self).__init__()
        self.order.append("param_init_fn")
        self.domains.append(param_init_fns)

    def initialize(self, in_d, scope):
        if len(in_d) == 3 and in_d[1] == 1:
            warnings.warn('Please check that is indeed a CNN output as input (--getting rid of one dimension--)')
            super(ClassificationAttention, self).initialize(in_d, scope)
        elif len(in_d) != 2:
            raise ValueError('Should be 2-dimensional (timesteps,dim)')
        else:
            super(ClassificationAttention, self).initialize(in_d, scope)

    def get_outdim(self):
        if len(self.in_d) == 3:
            dim = self.in_d[2]
        else:
            dim = self.in_d[1]
        return (dim, )

    def compile(self, in_x, train_feed, eval_feed):
        param_init_fn = self.domains[0][self.chosen[0]]
        if len(self.in_d) == 2:
            timesteps, dim = self.in_d
        else:
            in_x = tf.squeeze(in_x, axis=[2])
            timesteps, _ , dim = self.in_d

        W = tf.Variable(param_init_fn([dim, dim]))
        b = tf.Variable(tf.zeros([dim]))

        sc = np.sqrt(6.0)/np.sqrt(dim + 1)
        u_s = tf.Variable(tf.random_uniform([dim], -sc, sc))

        u = tf.add(tf.tensordot(in_x, W, axes=1), b)
        u = tf.tanh(u)
        u_dot_us = tf.tensordot(u, u_s, axes=1)

        alphas = tf.nn.softmax(u_dot_us)

        return tf.reduce_sum(in_x * tf.expand_dims(alphas, -1), 1)











