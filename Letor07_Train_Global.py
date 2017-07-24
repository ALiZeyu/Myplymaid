"""This is the Training Structure source code.

This module is the main function of model training.
usage:
    python Letor07_Train_Global.py [config_file]
"""

__version__ = '0.1'
__author__ = 'Liang Pang'

import sys
sys.path.insert(0, 'model/')

import json
config = json.loads( open(sys.argv[1]).read() )

import data0_utils as du
Letor07Path = config['data_dir'] 

config['fill_word'] = du._PAD_
config['embedding'] = du.embedding
config['feat_size'] = du.feat_size

# pair_gen = du.PairGenerator(rel_file=Letor07Path + '/relation.train.fold%d.txt'%(config['fold']), config=config)
pair_gen = du.PairGenerator(rel_file=Letor07Path + '/relation_train.txt', config=config)

from importlib import import_module
mo = import_module(config['model_file'])
model = mo.Model(config)

import numpy as np
import tensorflow as tf
import math

se_config = tf.ConfigProto()
se_config.gpu_options.allow_growth=True
sess = tf.Session(config=se_config)
model.init_step(sess)

flog = open(config['log_file'], 'w')
for i in range(config['epoch']):
    iter_num = int(math.ceil(pair_gen.get_rel_len()/config['batch_size']))
    loss_list = []
    for j in range(iter_num):
        X1, X1_len, X2, X2_len, Y, F = pair_gen.new_get_batch(j, data1=du.query_data, data2=du.doc_data)
        feed_dict={ model.X1: X1, model.X1_len: X1_len, model.X2: X2,
                    model.X2_len: X2_len, model.Y: Y, model.F: F}
        loss = model.train_step(sess, feed_dict)
        loss_list.append(loss)
    print >>flog, '[Train:%s]'%i, np.mean(loss_list)
    print '[Train:%s]'%i, np.mean(loss_list)
    flog.flush()

    model.saver.save(sess, 'checkpoint/%s.ckpt'%(config['model_tag']), global_step=i)
    list_gen = du.ListGenerator(rel_file=Letor07Path + '/relation_test.txt', config=config)
    X1, X1_len, X2, X2_len, Y, F = list_gen.get_batch(data1=du.query_data, data2=du.doc_data)
    feed_dict={ model.X1: X1, model.X1_len: X1_len, model.X2: X2,
                model.X2_len: X2_len, model.Y: Y, model.F: F}
    pred = model.test_step(sess, feed_dict)
    eval = np.mean(np.argmax(pred, 1) == np.argmax(Y, 1))
    print >>flog, '[Test:%s]'%i, eval
    print '[Test:%s]'%i, eval
    flog.flush()
flog.close()

