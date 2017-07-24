"""This is the Data Utils for Letor source code.

This module is used to read data from letor dataset.
"""

__version__ = '0.1'
__author__ = 'Liang Pang'

import sys
import random
import numpy as np
import math

import json
config = json.loads( open(sys.argv[1]).read() )

Letor07Path = config['data_dir'] #'/home/pangliang/matching/data/letor/r5w/'
import pytextnet as pt

word_dict, iword_dict = pt.io.base.read_word_dict(filename=Letor07Path + '/word_dict.txt')
# read_data : {index : sentence}
query_data = pt.io.base.read_data(filename=Letor07Path + '/docid.txt')
doc_data = pt.io.base.read_data(filename=Letor07Path + '/docid.txt')
embed_dict = pt.io.base.read_embedding(filename=Letor07Path + '/10003emb.npy')

feat_size = 0

'''
_PAD_ = len(word_dict)
embed_dict[_PAD_] = np.zeros((config['embed_size'], ), dtype=np.float32)
word_dict[_PAD_] = '[PAD]'
iword_dict['[PAD]'] = _PAD_

_UNK_ = len(word_dict)
embed_dict[_UNK_] = np.zeros((config['embed_size'], ), dtype=np.float32)
word_dict[_UNK_] = '[UNK]'
iword_dict['[UNK]'] = _UNK_
'''
_PAD_ = 0
word_dict[_PAD_] = '[PAD]'
iword_dict['[PAD]'] = _PAD_

_UNK_ = 10001
word_dict[_UNK_] = '[UNK]'
iword_dict['[UNK]'] = _UNK_

W_init_embed = np.float32(np.random.uniform(-0.02, 0.02, [len(word_dict), config['embed_size']]))
embedding = pt.io.base.convert_embed_2_numpy(embed_dict, embed = W_init_embed)

class PairGenerator():
    def __init__(self, rel_file, config):
        self.rel = pt.io.base.read_relation(filename=rel_file)
        # rel is a list of ((int)label, query_id, doc_id) tuple
        # self.pair_list = self.make_pair(self.rel)
        self.config = config

    def get_rel_len(self):
        return float(len(self.rel))
    #
    # def make_pair(self, rel):
    #     rel_set = {}
    #     pair_list = []
    #     for label, d1, d2 in rel:
    #         if d1 not in rel_set:
    #             rel_set[d1] = {}
    #         if label not in rel_set[d1]:
    #             rel_set[d1][label] = []
    #         rel_set[d1][label].append(d2)
    #     for d1 in rel_set:
    #         label_list = sorted(rel_set[d1].keys(), reverse = True)
    #         for hidx, high_label in enumerate(label_list[:-1]):
    #             for low_label in label_list[hidx+1:]:
    #                 for high_d2 in rel_set[d1][high_label]:
    #                     for low_d2 in rel_set[d1][low_label]:
    #                         pair_list.append( (d1, high_d2, low_d2) )
    #     print 'Pair Instance Count:', len(pair_list)
    #     return pair_list


    # data1 : query or doc dict the format is  {id : words index}
    # def get_batch(self, data1, data2):
    #     config = self.config
    #     X1 = np.zeros((config['batch_size']*2, config['data1_maxlen']), dtype=np.int32)
    #     X1_len = np.zeros((config['batch_size']*2,), dtype=np.int32)
    #     X2 = np.zeros((config['batch_size']*2, config['data2_maxlen']), dtype=np.int32)
    #     X2_len = np.zeros((config['batch_size']*2,), dtype=np.int32)
    #     Y = np.zeros((config['batch_size']*2,), dtype=np.int32)
    #     F = np.zeros((config['batch_size']*2, feat_size), dtype=np.float32)
    #     # 1 for every two position
    #     Y[::2] = 1
    #     X1[:] = config['fill_word']
    #     X2[:] = config['fill_word']
    #     for i in range(config['batch_size']):
    #         d1, d2p, d2n = random.choice(self.pair_list)
    #         d1_len = min(config['data1_maxlen'], len(data1[d1]))
    #         d2p_len = min(config['data2_maxlen'], len(data2[d2p]))
    #         d2n_len = min(config['data2_maxlen'], len(data2[d2n]))
    #         X1[i*2,   :d1_len],  X1_len[i*2]   = data1[d1][:d1_len],   d1_len
    #         X2[i*2,   :d2p_len], X2_len[i*2]   = data2[d2p][:d2p_len], d2p_len
    #         X1[i*2+1, :d1_len],  X1_len[i*2+1] = data1[d1][:d1_len],   d1_len
    #         X2[i*2+1, :d2n_len], X2_len[i*2+1] = data2[d2n][:d2n_len], d2n_len
    #         #F[i*2] = features[(d1, d2p)]
    #         #F[i*2+1] = features[(d1, d2n)]
    #
    #     return X1, X1_len, X2, X2_len, Y, F

    def new_get_batch(self, index, data1, data2):
        config = self.config
        X1 = np.zeros((config['batch_size'], config['data1_maxlen']), dtype=np.int32)
        X1_len = np.zeros((config['batch_size'],), dtype=np.int32)
        X2 = np.zeros((config['batch_size'], config['data2_maxlen']), dtype=np.int32)
        X2_len = np.zeros((config['batch_size'],), dtype=np.int32)
        Y = np.zeros((config['batch_size'], 2), dtype=np.int32)
        F = np.zeros((config['batch_size'], feat_size), dtype=np.float32)
        X1[:] = config['fill_word']
        X2[:] = config['fill_word']

        b = config['batch_size'] * index
        e = min(len(self.rel), config['batch_size'] * (index+1))
        batch_list = self.rel[b:e]
        while len(batch_list) < config['batch_size']:
            p = random.randint(0, len(self.rel) - 1)
            batch_list.append(self.rel[p])

        for i in range(config['batch_size']):
            label, d1, d2 = batch_list[i]
            d1_len = min(config['data1_maxlen'], len(data1[d1]))
            d2_len = min(config['data2_maxlen'], len(data2[d2]))
            X1[i, :d1_len], X1_len[i] = data1[d1][:d1_len], d1_len
            X2[i, :d2_len], X2_len[i] = data2[d2][:d2_len], d2_len
            Y[i] = [1,0] if label == 0 else [0,1]

        return X1, X1_len, X2, X2_len, Y, F



   
class ListGenerator():
    def __init__(self, rel_file, config):
        self.rel = pt.io.base.read_relation(filename=rel_file)
        # self.list_list = self.make_list(rel)
        self.config = config

    # def make_list(self, rel):
    #     list_list = {}
    #     for label, d1, d2 in rel:
    #         if d1 not in list_list:
    #             list_list[d1] = []
    #         list_list[d1].append( (label, d2) )
    #     for d1 in list_list:
    #         list_list[d1] = sorted(list_list[d1], reverse = True)
    #     print 'List Instance Count:', len(list_list)
    #     return list_list.items()

    def get_batch(self, data1, data2):
        config = self.config
        X1 = np.zeros((len(self.rel), config['data1_maxlen']), dtype=np.int32)
        X1_len = np.zeros((len(self.rel),), dtype=np.int32)
        X2 = np.zeros((len(self.rel), config['data2_maxlen']), dtype=np.int32)
        X2_len = np.zeros((len(self.rel),), dtype=np.int32)
        Y = np.zeros((len(self.rel),2), dtype= np.float32)
        F = np.zeros((len(self.rel), feat_size), dtype=np.float32)
        X1[:] = config['fill_word']
        X2[:] = config['fill_word']

        for i in range(len(self.rel)):
            label, d1, d2 = self.rel[i]
            d1_len = min(config['data1_maxlen'], len(data1[d1]))
            d2_len = min(config['data2_maxlen'], len(data2[d2]))
            X1[i, :d1_len], X1_len[i] = data1[d1][:d1_len], d1_len
            X2[i, :d2_len], X2_len[i] = data2[d2][:d2_len], d2_len
            Y[i] = [1, 0] if label == 0 else [0, 1]
            #F[j] = features[(d1, d2)]
        return X1, X1_len, X2, X2_len, Y, F
