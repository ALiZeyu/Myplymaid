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
def accuracy_cal(x1, x2):
    r = (np.argmax(x1,1) == np.argmax(x2,1))
    right = 0
    for i in range(len(r)):
        right += 1 if r[i] == True else 0
    return float(right)/len(r)

import tensorflow as tf

sess = tf.Session()
model.init_step(sess)

flog = open(config['log_file'], 'w')
for i in range(config['train_iters']):
    # the format of get_batch id batch * 2
    X1, X1_len, X2, X2_len, Y, F = pair_gen.new_get_batch(data1=du.query_data, data2=du.doc_data)
    feed_dict={ model.X1: X1, model.X1_len: X1_len, model.X2: X2,
                model.X2_len: X2_len, model.Y: Y, model.F: F}
    loss = model.train_step(sess, feed_dict)
    if (i+1)%100 == 0:
        print >>flog, '[Train:%s]'%i, loss
        print '[Train:%s]'%i, loss
    flog.flush()

    if i == 0:
        model.saver.save(sess, 'checkpoint/%s.ckpt'%(config['model_tag']), global_step=i)

    if (i+1) % 200 == 0:
        model.saver.save(sess, 'checkpoint/%s.ckpt'%(config['model_tag']), global_step=i)
        list_gen = du.ListGenerator(rel_file=Letor07Path + '/relation_test.txt', config=config)
        X1, X1_len, X2, X2_len, Y, F = list_gen.get_batch(data1=du.query_data, data2=du.doc_data)
        feed_dict={ model.X1: X1, model.X1_len: X1_len, model.X2: X2,
                    model.X2_len: X2_len, model.Y: Y, model.F: F}
        pred = model.test_step(sess, feed_dict)
        # correct_prediction = tf.equal(tf.argmax(model.Y, 1), tf.argmax(pred, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        eval = accuracy_cal(pred, Y)
        print >>flog, '[Test:%s]'%i, eval
        print '[Test:%s]'%i, eval
        flog.flush()
flog.close()

