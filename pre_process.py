import codecs
import numpy as np


def read_file(path):
    f = codecs.open(path, 'r', encoding='utf-8')
    content = f.readlines()
    content = [sen.strip() for sen in content]
    f.close()
    return content


def write_file(path, content):
    f = codecs.open(path, 'w', encoding='utf-8')
    for sentence in content:
        f.write(sentence+'\n')
    f.close()

# convert dict format from dssm to pyramid
def dict_trans():
    dssm_dict = read_file('data/vocaburary_chatlog1.txt')
    result = []
    index = 0
    for word in dssm_dict:
        result.append(word +  ' ' +str(index))
        index += 1
    write_file('data/word_dict.txt', result)


# generate relation file and qid,docid
# docid : id length words
def train_test_trans(dict_file, dssm_file):
    word_file = read_file(dict_file)
    word_dict = dict()
    for sen in word_file:
        word_dict[sen.split()[0]] = sen.split()[1]
    dssm_data = read_file(dssm_file)
    docid = []
    relation = []
    index = 0
    doc_dict = dict()
    for sen in dssm_data:
        doc1 = sen.split('\t')[0]
        doc2 = sen.split('\t')[1]
        rela = sen.split('\t')[2]
        doc1_str = trans_sen_to_id_array(word_dict, doc1)
        if doc1_str not in doc_dict:
            doc_dict[doc1_str] = index
            docid.append(str(index)+' '+doc1_str)
            index += 1
        doc2_str = trans_sen_to_id_array(word_dict, doc2)
        if doc2_str not in doc_dict:
            doc_dict[doc2_str] = index
            docid.append(str(index) + ' ' + doc2_str)
            index += 1
        relation.append(str(rela)+' '+str(doc_dict[doc1_str])+' '+str(doc_dict[doc2_str]))
    write_file('data/relation_all.txt', relation)
    write_file('data/docid.txt', docid)


def trans_sen_to_id_array(word_dict,doc):
    doc_array = doc.split()
    doc_str = str(len(doc_array))
    for word in doc_array:
        num = word_dict[word] if word in word_dict else len(word_dict) + 1
        doc_str += ' '+str(num)
    return doc_str.strip()

def accuracy_cal():
    x1=[[1,0],[0,1],[1,0]]
    x2 = [[0.8,0.2],[0.3,0.7],[0.2,0.8]]
    r = (np.argmax(x1,1) == np.argmax(x2,1))
    right = 0
    for i in range(len(r)):
        right += 1 if r[i] == True else 0
    print float(right)/len(r)


if __name__ == '__main__':
    #train_test_trans('data/word_dict.txt','data/train_dssm.txt')
    accuracy_cal()