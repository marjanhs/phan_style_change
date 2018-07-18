
import os, math
from nltk import word_tokenize
import torch, json

soh = "\x02"
epsilon = 1e-04


def read_file(file_path):
    '''
    needs to be modified
    :param file_path:
    :return:
    '''
    with open(file_path) as f:
        all=f.read()
        all=all.encode('utf-8')
        all=all.decode('utf-8')
    return all


def load_PAN_labels(label_files):
    labels = {}
    with open(label_files,'r') as f:
        for line in f:
            doc_id, lbl = line.split()
            labels[doc_id.strip()] = '1' if lbl.strip() == 'Y' else '0'
    return labels


def get_vocabulary(collection):
    vocabulary = set()
    for doc in collection:
        words = word_tokenize(doc)
        vocabulary.update(words)
    return vocabulary



def softmax(vector):
    upper = [math.exp(v) for v in vector]
    sum_upper = sum(upper)
    return [u/sum_upper for u in upper]

def euclidean_distance(x1, x2, dim=1):
    r"""Returns Euclidean distance between x1 and x2, computed along dim.
        Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
    """
    return (torch.sqrt(torch.sum((x1-x2) ** 2, dim))).squeeze()


def dot(x1, x2, dim=1):
    return torch.sum(x1 * x2, dim).squeeze()


def mean_of_l1(x1, x2, dim=1):
    return torch.mean(torch.abs(x1 - x2), dim).squeeze()


def sigmoid_kernel(x1, x2, dim=1, gamma=None, c=1):
    if gamma is None:
        gamma = 1.0/x1.size()[dim]
    output = torch.tanh(gamma * dot(x1, x2, dim)+c)
    return output


def chi_squared(x1, x2, dim=1, gamma=1, eps=1e-8):
    return torch.exp(- gamma * torch.sum(((x1 - x2) ** 2)/(x1 + x2).clamp(min=eps), dim))


def rbf_kernel(x1, x2, dim=1, gamma=1):
    output = torch.sum((x1 - x2) ** 2, dim)
    return torch.exp(- gamma * output)


def print_json(json_dic, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for k,v in json_dic.items():
        data={}
        data["changes"] = v
        local_path = os.path.join(path, str(k)+'.truth')
        with open(local_path, 'w') as fw:
            json.dump(data, fw)
