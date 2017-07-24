"""This is the Model File of MatchPyramid.

This module is used to construct the MatchPyramid described in paper https://arxiv.org/abs/1602.06359.
"""

__version__ = '0.1'
__author__ = 'Liang Pang'

import sys

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib.rnn.python.ops import rnn

"""
Model Class
"""
class Model():

    def __init__(self, config):
        self.config = config
        tf.reset_default_graph()
        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, config['data1_maxlen']))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, config['data2_maxlen']))
        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None, ))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None, ))
        self.Y = tf.placeholder(tf.float32, name='Y', shape=(None, 2))
        self.F = tf.placeholder(tf.float32, name='F', shape=(None, config['feat_size']))

        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index', shape=(None, config['data1_maxlen'], config['data2_maxlen'], 3))

        self.batch_size = tf.shape(self.X1)[0]
        
        self.embedding = tf.get_variable('embedding', initializer = config['embedding'], dtype=tf.float32, trainable=False)

        self.embed1 = tf.nn.embedding_lookup(self.embedding, self.X1)
        self.embed2 = tf.nn.embedding_lookup(self.embedding, self.X2)
        # print 'embedding'
        # print self.embed1.get_shape()
        '''        
        bi_outputs1, state_fw, state_bw = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[rnn_cell.GRUCell(config['embed_size'])],
            cells_bw=[rnn_cell.GRUCell(config['embed_size'])],
            inputs=self.embed1,
            dtype=tf.float32,
            sequence_length=self.X1_len)

        bi_outputs2, state_fw, state_bw = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[rnn_cell.GRUCell(config['embed_size'])],
            cells_bw=[rnn_cell.GRUCell(config['embed_size'])],
            inputs=self.embed2,
            dtype=tf.float32,
            sequence_length=self.X2_len)
        
        bi_outputs1 = self.bi_rnn_encode(config['embed_size'], self.embed1, self.X1_len)
        bi_outputs2 = self.bi_rnn_encode(config['embed_size'], self.embed2, self.X2_len)

        '''
        #with tf.variable_scope("QEncoder",initializer=tf.uniform_unit_scaling_initializer(1.0), reuse=False) as Qencoder:
        bi_outputs1 = self.bi_rnn_encode(config['embed_size'], self.embed1, self.X1_len, '0')
        #with tf.variable_scope("DEncoder",initializer=tf.uniform_unit_scaling_initializer(1.0), reuse=False) as Dencoder:
        bi_outputs2 = self.bi_rnn_encode(config['embed_size'], self.embed2, self.X2_len, '1')
        # self.cross = tf.einsum('abd,acd->abc', self.embed1, self.embed2)
        self.cross = tf.einsum('abd,acd->abc', bi_outputs1, bi_outputs2)
        # batch_size * X1_maxlen * X2_maxlen
        # print self.cross.get_shape()
        self.cross_img = tf.expand_dims(self.cross, 3)
        # print self.cross_img.get_shape()
        
        # convolution
        self.w1 = tf.get_variable('w1', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32) , dtype=tf.float32, shape=[2, 2, 1, 8])
        self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer() , dtype=tf.float32, shape=[8])
        # batch_size * X1_maxlen * X2_maxlen * feat_out
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)
        # print self.conv1.get_shape()

        # dynamic pooling
        self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)
        # print self.conv1_expand.get_shape()
        self.pool1 = tf.nn.max_pool(self.conv1_expand, 
                        [1, config['data1_maxlen'] / config['data1_psize'], config['data2_maxlen'] / config['data2_psize'], 1], 
                        [1, config['data1_maxlen'] / config['data1_psize'], config['data2_maxlen'] / config['data2_psize'], 1], "VALID")


        with tf.variable_scope('fc1'):
            self.fc1 = tf.nn.relu(tf.contrib.layers.linear(tf.reshape(self.pool1, [self.batch_size, config['data1_psize'] * config['data2_psize'] * 8]), 20))

        #linear(inputs, num_outputs). Create a layer of num_outputs nodes fully connected to the previous layer second_hidden_layer with no activation function,
        # just a linear transformation:
        # self.pred = tf.contrib.layers.linear(self.fc1, 1)
        # p_pred = tf.Print(self.pred, [self.pred, tf.shape(self.pred),'pred'])
        #
        # # pos = tf.strided_slice(self.pred, [0], [self.batch_size], [2])
        # # neg = tf.strided_slice(self.pred, [1], [self.batch_size], [2])
        # pos = tf.strided_slice(p_pred, [0], [self.batch_size], [2])
        # neg = tf.strided_slice(p_pred, [1], [self.batch_size], [2])
        #
        # p_neg = tf.Print(neg, [neg, tf.shape(neg), 'neg'])
        #
        # self.loss = tf.reduce_mean(tf.maximum(1.0 + p_neg - pos, 0.0))
        self.pred = tf.contrib.layers.linear(self.fc1, 2)
        self.prob = tf.nn.softmax(self.pred)
        # p_pred = tf.Print(self.pred,[self.pred, tf.shape(self.pred)])
        # p_Y = tf.Print(self.Y,[self.Y, tf.shape(self.Y)])
        self.loss = -tf.reduce_sum(self.Y * tf.log(self.prob))
        correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.train_model = tf.train.AdamOptimizer().minimize(self.loss)
    
        self.saver = tf.train.Saver(max_to_keep=5)

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
        return np.array(index)

    '''
    def bi_rnn_encode(self, hidden_size, emb, X_len):
        #with tf.variable_scope("Encoder", reuse=True):
            #with tf.variable_scope('forward', reuse=True):
        self.GRU_fw_cell = rnn_cell.GRUCell(hidden_size)
            #with tf.variable_scope('backward', reuse=True):
        self.GRU_bw_cell = rnn_cell.GRUCell(hidden_size)
        out, fw_state, bw_state = rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=[self.GRU_fw_cell],
        cells_bw=[self.GRU_bw_cell],
        inputs=emb,
        dtype=tf.float32,
        sequence_length=X_len)
        return out
    '''
    def bi_rnn_encode(self, hidden_size, emb, X_len, scope_num):
        with tf.variable_scope("Encoder"+scope_num):
            #with tf.variable_scope('forward', reuse=True):
            self.GRU_fw_cell = rnn_cell.GRUCell(hidden_size)
                #with tf.variable_scope('backward', reuse=True):
            self.GRU_bw_cell = rnn_cell.GRUCell(hidden_size)
            out, fw_state, bw_state = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[self.GRU_fw_cell],
            cells_bw=[self.GRU_bw_cell],
            inputs=emb,
            dtype=tf.float32,
            sequence_length=X_len)
            return out

    def init_step(self, sess):
        sess.run(tf.global_variables_initializer())

    def train_step(self, sess, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len], 
                                            self.config['data1_maxlen'], self.config['data2_maxlen'])
        # _, loss = sess.run([self.train_model, self.loss], feed_dict=feed_dict)
        _, accuracy = sess.run([self.train_model, self.accuracy], feed_dict=feed_dict)
        return accuracy

    def test_step(self, sess, feed_dict):
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len], 
                                            self.config['data1_maxlen'], self.config['data2_maxlen'])
        prob = sess.run(self.prob, feed_dict=feed_dict)
        return prob
    
    # def eval_step(self, sess, node, feed_dict):
    #     feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len], feed_dict[self.X2_len],
    #                                         self.config['data1_maxlen'], self.config['data2_maxlen'])
    #     node_value = sess.run(node, feed_dict=feed_dict)
    #     return node_value
