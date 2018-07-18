from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os, json
soh = "\x02"  # pt feature separator
US = "\x1f"  # unit separator => sentence separator
eot = '\x04'
arrow = ' => '
from tqdm import tqdm
from sypt_ptree import *

class PAN_Dataset(Dataset):
    def __init__(self, csv_file, limit=None):
        '''
        :param suffix_csv_file:  2015.csv for train2015.csv
        :return:
        '''

        self.dataset = pd.read_csv(csv_file, sep='\t')

        if limit is not None:
            self.dataset = self.dataset[:limit]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        doc = self.dataset.context[idx]
        id_ = self.dataset.id[idx]
        label = self.dataset.label[idx]
        return {'id': id_, 'doc': doc,  'label': label}

    def getlabels(self, indices):
        return list(self.dataset.label[indices])


def get_word_index(word_list):
    '''
    :param data: list of words
    :return: word_index
    '''
    word_to_index = {}
    index_to_word ={}
    for word in word_list:
        if word not in word_to_index:
            idx = len(word_to_index)
            word_to_index[word] = idx
            index_to_word[idx] = word
    return word_to_index, index_to_word


def get_embedding_matrix(embedding_index, word_index, embd_dim):
    embedding_matrix = np.random.random((len(word_index), embd_dim))
    for word, i in word_index.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
    return embedding_matrix


def get_embedding_index(embedding_file):
    embeddings_index = {}
    f = open(embedding_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def get_pt_embedding_index(pt_features, embedding_dim):
    embeddings_index = {}
    for pt in pt_features:
        if pt not in embeddings_index.keys() and pt != '':
            embeddings_index[pt] = np.asarray(np.random.rand(1, embedding_dim), dtype='float32')
    return embeddings_index


def load_dataset_and_pt_embedding(pt_files=None, embedding_dim=100):
    '''

    :param csv_files:
    :param embedding_dim:
    :return:
    '''

    all = []
    limit = None
    datasets = dict()
    for ds_name, ds_file in pt_files.items():
        d = PAN_Dataset(ds_file, limit)
        datasets[ds_name] = d
        for idx in range(len(d)):
            doc_pt = d[idx]["doc"].split(US)
            all.extend(sent.split(soh) for sent in doc_pt)

    all = [item for sublist in all for item in sublist] #?
    word_to_index, index_to_word = get_word_index(all)
    embedding_index = get_pt_embedding_index(all, embedding_dim)
    embd_matrix = get_embedding_matrix(embedding_index, word_to_index, embedding_dim)
    return datasets, word_to_index, embd_matrix, index_to_word


def getclassindices(dataset):
    pos = []
    neg = []
    for i in range(len(dataset.dataset)):
        if dataset.dataset.label[i] == 0:
            pos.append(i)
        elif dataset.dataset.label[i] == 1:
            neg.append(i)
    assert  len(pos) == len(neg)
    return pos, neg


def get_label_pan2018(json_path):
    data = json.load(open(json_path))
    label = data["changes"]
    if label:
        label = 1
    elif not label:
        label = 0
    else:
        raise ValueError('The label value is not True/False')
    return str(label)


def create_csv_pan2018(source_path, dest_path):
    s = 0
    with open(dest_path, 'w', encoding = 'utf-8') as fw:
        fw.writelines('id\tlabel\tcontext\n')
        for file in os.listdir(source_path):
            if file.endswith('.txt'):
                name = file.replace('.txt', '')
                with open(os.path.join(source_path, file), 'r', encoding='utf-8') as f:
                    context = f.read().replace('\n', '')
                truth = os.path.join(source_path,name+'.truth')
                if os.path.exists(truth):
                    label = get_label_pan2018(truth)
                else:
                    label = "0"
                fw.write(name+'\t'+label+'\t'+context+'\n')
                s += 1
    print('%d files were read and collected' % s)


def create_pt_pan2018(source_path, dest_path, root='../../../../', server_type='standalone', tmp_path=''):
    dataset = PAN_Dataset(source_path, None)
    total = len(dataset)
    pbar = tqdm(total=total)
    with open(dest_path, 'w') as fw:
        fw.writelines('id\tlabel\tcontext\n')
        for idx in range(len(dataset)):
            pbar.update(1)
            d = dataset[idx]
            context = d["doc"]
            try:
                if server_type == 'standalone':
                    pt = get_pt_features_standalone(context, root, tmp_path)
                elif server_type == 'corenlp':
                    pt = get_pt_features_coreNLP(context)
                else:
                    print("server type is not correct! %s" % server_type)
            except:
                raise Exception('\n problem in parsing %d id = %s .\n' % (idx, d["id"]))
            flatted_pt = ''
            for sentence in pt:
                for feature in sentence:
                    flatted_pt += eot.join(feature) + soh
                flatted_pt += US
            fw.write(d["id"] + '\t' + str(d["label"]) + '\t' + flatted_pt + '\n')


