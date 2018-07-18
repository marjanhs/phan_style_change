import torch, os, argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sypt_dataset, sypt_utils
from torch.utils.data import DataLoader
from sypt_utils import *
from sypt_dataset import create_pt_pan2018
US = "\x1f"  # unit separator => sentence separator
soh = "\x02"


class PTFAttenPRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_martix, batch_size, iscuda= True):
        super(PTFAttenPRNN, self).__init__()
        self.batch_size = batch_size
        self.ptf_hidden_size = hidden_dim
        self.ptf_embed_dim = embedding_dim
        self.iscuda = iscuda

        self.ptf_embed = nn.Embedding(vocab_size, embedding_dim)
        self.ptf_embed.weight.data.copy_(torch.from_numpy(embedding_martix))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.ptf_context_vector = self.init_ptf_contx_vector()
        self.ptf_hidden = self.init_ptf_hidden()
        self.lin_attention = nn.Linear(self.ptf_hidden_size, self.ptf_hidden_size)

    def init_ptf_hidden(self):
        if self.iscuda:
            return Variable(torch.zeros(1, self.batch_size, self.ptf_hidden_size)).cuda(),\
               Variable(torch.zeros(1, self.batch_size, self.ptf_hidden_size)).cuda()
        else:
            return Variable(torch.zeros(1, self.batch_size, self.ptf_hidden_size)), \
                   Variable(torch.zeros(1, self.batch_size, self.ptf_hidden_size))

    def init_ptf_contx_vector(self):
        return nn.Parameter(torch.Tensor(self.ptf_hidden_size, 1))

    def get_ptf_attention(self, ptf_encoded):
        u = F.tanh(self.lin_attention(ptf_encoded))
        mul = torch.matmul(u, self.ptf_context_vector.squeeze())
        assert mul.size() == torch.Size([ptf_encoded.size(0), self.batch_size])
        alpha = F.softmax(mul, dim=0).unsqueeze(2)# (seq_length, batch_size)->(seq_length,batch_size,1)
        return alpha * ptf_encoded

    def forward(self,  ptf_sequence, ptf_hidden_state):
        embeded_words = self.ptf_embed(ptf_sequence).view(len(ptf_sequence), self.batch_size, -1)
        ptf_output, ptf_hidden_state = self.lstm(embeded_words, ptf_hidden_state)
        ptf_attention = self.get_ptf_attention(ptf_output)
        s_i = torch.sum(ptf_attention, dim=0).unsqueeze(0)
        return s_i, ptf_hidden_state


class PTSentAttenRNN(nn.Module):
    def __init__(self, batch_size, sent_hidden_size, ptf_hidden_size, class_no, drop_rate, iscuda=True, fuse=True):
        super(PTSentAttenRNN, self).__init__()
        self.batch_size = batch_size
        self.ptf_hidden_size = ptf_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.drop_rate = drop_rate
        self.iscuda = iscuda
        self.fuse = fuse

        self.sent_lstm_l = nn.LSTM(ptf_hidden_size, sent_hidden_size)
        self.sent_context_vector_l = self.init_sent_contx_vector()
        self.sent_hidden_l = self.init_sent_hidden()
        self.lin_attention_l = nn.Linear(self.sent_hidden_size, self.sent_hidden_size)

        self.sent_lstm_r = nn.LSTM(ptf_hidden_size, sent_hidden_size)
        self.sent_context_vector_r = self.init_sent_contx_vector()
        self.sent_hidden_r = self.init_sent_hidden()
        self.lin_attention_r = nn.Linear(self.sent_hidden_size, self.sent_hidden_size)
        self.lin = nn.Linear(7, class_no) if self.fuse else nn.Linear(2*self.sent_hidden_size, class_no)

    def forward(self, ptf_atten_sequence, sent_hidden_state):
        ptf_atten_seq_l, ptf_atten_seq_r = ptf_atten_sequence[0], ptf_atten_sequence[1]
        sent_hidden_state_l, sent_hidden_state_r = sent_hidden_state[0], sent_hidden_state[1]

        sent_output_l, sent_hidden_state_l = self.sent_lstm_l(ptf_atten_seq_l, sent_hidden_state_l)
        sent_attention_l = self.get_sent_attention_l(sent_output_l)
        l_hidden = torch.sum(sent_attention_l, dim=0)

        sent_output_r, sent_hidden_state_r = self.sent_lstm_r(ptf_atten_seq_r, sent_hidden_state_r)
        sent_attention_r = self.get_sent_attention_r(sent_output_r)
        r_hidden = torch.sum(sent_attention_r, dim=0)
        
        sent_hidden_state = [sent_hidden_state_l, sent_hidden_state_r]
        merged = PTSentAttenRNN.get_last_layer(l_hidden, r_hidden, self.fuse)
        merged = F.dropout(merged, p=self.drop_rate, training= self.training)
        merged = self.lin(merged)
        return F.log_softmax(merged, dim=1), sent_hidden_state

    def get_sent_attention_l(self, sent_encoded):
        u = F.tanh(self.lin_attention_l(sent_encoded))
        mul = torch.matmul(u, self.sent_context_vector_l.squeeze())
        assert mul.size() == torch.Size([sent_encoded.size(0), self.batch_size])
        alpha = F.softmax(mul, dim=0).unsqueeze(2) # (sent_no, batch_size)->(sent_no,batch_size,1)
        return alpha * sent_encoded

    def get_sent_attention_r(self, sent_encoded):
        u = F.tanh(self.lin_attention_r(sent_encoded))
        mul = torch.matmul(u, self.sent_context_vector_r.squeeze())
        assert mul.size() == torch.Size([sent_encoded.size(0), self.batch_size])
        alpha = F.softmax(mul, dim=0).unsqueeze(2) # (sent_no, batch_size)->(sent_no,batch_size,1)
        return alpha * sent_encoded

    def init_sent_contx_vector(self):
        return nn.Parameter(torch.Tensor(self.sent_hidden_size, 1))
    @staticmethod
    def get_last_layer(l_hidden, r_hidden, fuse=True):
        if fuse:
            cos = F.cosine_similarity(l_hidden, r_hidden, dim=1).view(1, -1)
            euc = sypt_utils.euclidean_distance(l_hidden, r_hidden, dim=1).view(1, -1)
            dot_dis = sypt_utils.dot(l_hidden, r_hidden, dim=1).view(1, -1)
            mean_l1 = sypt_utils.mean_of_l1(l_hidden, r_hidden, dim=1).view(1, -1)
            sig = sypt_utils.sigmoid_kernel(l_hidden, r_hidden, dim=1).view(1, -1)
            chi = sypt_utils.chi_squared(l_hidden, r_hidden, dim=1).view(1, -1)
            rbf = sypt_utils.rbf_kernel(l_hidden, r_hidden, dim=1).view(1, -1)
            return torch.cat([cos, euc, dot_dis, mean_l1, sig, chi, rbf], dim=0).view(1, -1)
        else:
            return torch.cat([l_hidden, r_hidden], dim=1).view(1, -1)

    def init_sent_hidden(self):
        if self.iscuda:
            return Variable(torch.zeros(1, self.batch_size, self.sent_hidden_size)).cuda(),\
                   Variable(torch.zeros(1, self.batch_size, self.sent_hidden_size)).cuda()
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_hidden_size)),\
                   Variable(torch.zeros(1, self.batch_size, self.sent_hidden_size))


def make_context_vector(context, ptf_index): # ok
    return [ptf_index[word] for word in context if word in ptf_index]


def train_data(x_train, y_target, ptf_attn_model, sent_attn_model, ptf_optimizer, sent_optimizer, criterion):
    ptf_attn_model_l, ptf_attn_model_r = ptf_attn_model[0], ptf_attn_model[1]
    ptf_optimizer_l, ptf_optimizer_r = ptf_optimizer[0], ptf_optimizer[1]
    state_ptf_l, state_ptf_r = ptf_attn_model_l.init_ptf_hidden(), ptf_attn_model_r.init_ptf_hidden()
    state_sent = [sent_attn_model.init_sent_hidden(), sent_attn_model.init_sent_hidden()]

    y_target = Variable(torch.LongTensor(y_target))
    ptf_optimizer_l.zero_grad()
    ptf_optimizer_r.zero_grad()
    sent_optimizer.zero_grad()
    s_l, s_r = None, None

    for i in range(len(x_train[0])):
        ptf_idx_seq = Variable(torch.LongTensor(x_train[0][i])).cuda()
        _s, state_ptf_l = ptf_attn_model_l(ptf_idx_seq, state_ptf_l)
        if s_l is None:
            s_l = _s
        else:
            s_l = torch.cat((s_l, _s), 0)
    assert len(x_train[0]) == len(s_l)

    for i in range(len(x_train[1])):
        ptf_idx_seq = Variable(torch.LongTensor(x_train[1][i])).cuda()
        _s, state_ptf_r = ptf_attn_model_r(ptf_idx_seq, state_ptf_r)
        if s_r is None:
            s_r = _s
        else:
            s_r = torch.cat((s_r, _s), 0)
    assert len(x_train[1]) == len(s_r)

    y_pred, state_sent = sent_attn_model([s_l, s_r], state_sent)
    loss_train = criterion(y_pred.cuda(), y_target.cuda())
    loss_train.backward()

    ptf_optimizer_l.step()
    ptf_optimizer_r.step()
    sent_optimizer.step()
    return loss_train.data.item()


def tst_data(x_test, y_target, ptf_attn_model, sent_attn_model, criterion, iscuda):
    ptf_attn_model_l, ptf_attn_model_r = ptf_attn_model[0], ptf_attn_model[1]
    state_ptf_l, state_ptf_r = ptf_attn_model_l.init_ptf_hidden(), ptf_attn_model_r.init_ptf_hidden()
    state_sent = [sent_attn_model.init_sent_hidden(), sent_attn_model.init_sent_hidden()]
    s_l, s_r = None, None

    for i in range(len(x_test[0])):

        ptf_idx_seq = Variable(torch.LongTensor(x_test[0][i]))
        if iscuda:
            ptf_idx_seq = ptf_idx_seq.cuda()

        _s, state_ptf_l = ptf_attn_model_l(ptf_idx_seq, state_ptf_l)
        if s_l is None:
            s_l = _s
        else:
            s_l = torch.cat((s_l, _s), 0)
    assert len(x_test[0]) == len(s_l)

    for i in range(len(x_test[1])):
        ptf_idx_seq = Variable(torch.LongTensor(x_test[1][i]))
        if iscuda:
            ptf_idx_seq = ptf_idx_seq.cuda()
        _s, state_ptf_r = ptf_attn_model_r(ptf_idx_seq, state_ptf_r)
        if s_r is None:
            s_r = _s
        else:
            s_r = torch.cat((s_r, _s), 0)
    assert len(x_test[1]) == len(s_r)

    y_pred, state_sent = sent_attn_model([s_l, s_r], state_sent)
    if iscuda:
        loss_test = criterion(y_pred.cuda(), y_target.cuda())
    else:
        loss_test = criterion(y_pred, y_target)
    return y_pred, loss_test.data.item()


def eval(dataloader, ptf_index, criterion, return_json=False, models=None, iscuda=True):
    for mdl in models.values():
        mdl.eval()

    ptf_model_l = models["ptf_model_l"]
    ptf_model_r = models["ptf_model_r"]
    sent_model = models["sent_model"]

    total, correct = 0, 0
    total_loss = torch.Tensor([0])
    if iscuda:
        total_loss = total_loss.cuda()

    if return_json:
        json={}
    for itr, d in enumerate(dataloader):
        l_doc = d["doc"][0]
        l_doc = l_doc.split(US)
        target = d["label"]

        l_vec = []
        for e in l_doc:
            cv = make_context_vector(e.split(soh), ptf_index)
            if len(cv) != 0:
                l_vec.append(cv)
        r_vec = backward(l_vec)
        l_vec = list_of_list_to_long_tensor(l_vec)
        r_vec = list_of_list_to_long_tensor(r_vec)
        target = Variable(torch.LongTensor(target))

        if iscuda:
            target = target.cuda()
        data_test = [l_vec, r_vec]
        ptf_model = [ptf_model_l, ptf_model_r]
        outputs, loss = tst_data(data_test, target,  ptf_model, sent_model, criterion, iscuda)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        if return_json:
            json[d["id"][0]] = bool(predicted.cpu().numpy()[0])
        correct += (predicted == target.data).sum()
        total_loss += loss
    if return_json:
        return (100 * correct / total), (total_loss/len(dataloader))[0], json
    else:
        return (100 * correct / total), (total_loss / len(dataloader))[0]


def backward(doc):
    rdoc = list(reversed(doc))
    return [list(reversed(e)) for e in rdoc]


def list_of_list_to_long_tensor(src_list):
    des_list = [torch.LongTensor(e) for e in src_list]
    return des_list


def train_epoch(dataloader, ptf_index, models, optmzrs, loss_func):
    ptf_optim_l = optmzrs["ptf_optim_l"]
    ptf_optim_r = optmzrs["ptf_optim_r"]
    sent_optim = optmzrs["sent_optim"]
    for mdl in models.values():
        mdl.train()
    ptf_model_l = models["ptf_model_l"]
    ptf_model_r = models["ptf_model_r"]
    sent_model = models["sent_model"]
    total_loss = torch.Tensor([0]).cuda()
    for itr, d in enumerate(dataloader):

        l_doc = d["doc"][0]
        l_doc = l_doc.split(US)

        l_vec = []
        for e in l_doc:
            cv = make_context_vector(e.split(soh), ptf_index)
            if len(cv) != 0:
                l_vec.append(cv)
        r_vec = backward(l_vec)
        l_vec = list_of_list_to_long_tensor(l_vec)
        r_vec = list_of_list_to_long_tensor(r_vec)
        x_train = [l_vec, r_vec]
        ptf_model = [ptf_model_l, ptf_model_r]
        ptf_optim = [ptf_optim_l, ptf_optim_r]
        loss = train_data(x_train, d["label"], ptf_model, sent_model, ptf_optim, sent_optim, loss_func)
        total_loss += loss
    return (total_loss/len(dataloader))[0]


def get_params():
    params = dict()
    params["EMBEDDING_DIM"] = 100
    params["ptf_HIDDEN_DIM"] = 8
    params["SENT_HIDDEN_DIM"] = 8
    params["EPOCHS"] = 1
    params["dropout_rate"] = 0.3
    params["CLASS_NO"] = 2
    params["fuse"] = True
    params["iscuda"] = True

    return params


def save_checkpoint(models, is_best, model_name):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(models['ptf_model_l'].state_dict(), 'ptf_model_l' + model_name)
        torch.save(models['ptf_model_r'].state_dict(), 'ptf_model_r' + model_name)
        torch.save(models['sent_model'].state_dict(), 'sent_model' + model_name)
    else:
        print ("=> Validation Accuracy did not improve")


def train_model(train_path, val_path, model_name):
    '''
    train the model.

    :param train_path:
    :param val_path:
    :param model_name:
    :return:
    '''

    params = get_params()
    EMBEDDING_DIM = params["EMBEDDING_DIM"]
    ptf_HIDDEN_DIM = params["ptf_HIDDEN_DIM"]
    SENT_HIDDEN_DIM = params["SENT_HIDDEN_DIM"]
    EPOCHS = params["EPOCHS"]
    dropout_rate = params["dropout_rate"]
    batch_size = 1  # code should change a bit for batch size > 1
    CLASS_NO = params["CLASS_NO"]
    fuse = params["fuse"]

    for p,v in params.items():
        print('param %s = %s' % (p, str(v)))

    ds_files = dict()
    ds_files['train'] = train_path
    datasets, ptf_index, embd_matrix, index_word = sypt_dataset.load_dataset_and_pt_embedding\
        (ds_files, EMBEDDING_DIM)
    datasets["val"] = sypt_dataset.PAN_Dataset(val_path, None)
    train_dataloader = DataLoader(datasets["train"], 1, True)
    val_dataloader = DataLoader(datasets["val"], 1, True)
    VOCAB_SIZE = len(ptf_index)
    print('Vocab Size %d' % VOCAB_SIZE)
    print('train = %s , val = %s' % (train_path, val_path))

    # model definition
    ptf_model_l = PTFAttenPRNN(VOCAB_SIZE, EMBEDDING_DIM, ptf_HIDDEN_DIM, embd_matrix, batch_size).cuda()
    ptf_model_r = PTFAttenPRNN(VOCAB_SIZE, EMBEDDING_DIM, ptf_HIDDEN_DIM, embd_matrix, batch_size).cuda()
    pt_sent_model = PTSentAttenRNN(batch_size, SENT_HIDDEN_DIM, ptf_HIDDEN_DIM, CLASS_NO, dropout_rate, True, fuse=fuse).cuda()
    models = dict()
    models["ptf_model_l"] = ptf_model_l
    models["ptf_model_r"] = ptf_model_r
    models["sent_model"] = pt_sent_model

    # optimizers
    ptf_optim_l = optim.RMSprop(ptf_model_l.parameters(), lr=1e-03)
    ptf_optim_r = optim.RMSprop(ptf_model_r.parameters(), lr=1e-03)
    sent_optim = optim.RMSprop(pt_sent_model.parameters(), lr=1e-03)
    optmzrs = dict()
    optmzrs["ptf_optim_l"] = ptf_optim_l
    optmzrs["ptf_optim_r"] = ptf_optim_r
    optmzrs["sent_optim"] = sent_optim

    # loss function
    loss_func = nn.NLLLoss()

    # training and evaluation
    best_accuracy = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(train_dataloader, ptf_index, models, optmzrs, loss_func)
        val_acc, val_loss = eval(val_dataloader, ptf_index, loss_func, False, models)
        print('Epoch: %d and  train loss: %.4F val loss: %.4f  val acc: %.4F' % (epoch, train_loss, val_loss, val_acc))
        # Get bool not ByteTensor
        is_best = bool(val_acc > best_accuracy)
        # Get greater Tensor to keep track best acc
        best_accuracy = max(val_acc, best_accuracy)
        # Save checkpoint if is a new best
        save_checkpoint(models, is_best, model_name)

    # show the final results
    train_acc, train_loss = eval(train_dataloader, ptf_index, loss_func, False, models)
    print('train acc: %.4F train loss: %.10f ' % (train_acc, train_loss))

    val_acc, val_loss= eval(val_dataloader, ptf_index, loss_func, False, models)
    print('val acc: %.4F val loss: %.10f ' % (val_acc, val_loss))


def get_args():
    '''
    get arguments from command line
    :return: a dic of all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action='store', default='data/', help='source path')
    parser.add_argument('-o', action='store', default='data/', help='destination path')

    results = parser.parse_args()
    print(results)
    return vars(results)


if __name__ == "__main__":

    # param setting
    params = get_args()
    csv_path = params["c"]
    pt_path = params["o"]
    model_name = ''
    server = 'corenlp'

    train = f'{pt_path}train.{server}.pt'
    val = f'{pt_path}val.{server}.pt'

    train_csv = f'{csv_path}train.csv'
    val_csv = f'{csv_path}val.csv'

    # create ptf of train and val dataset
    if not os.path.exists(pt_path):
        os.mkdir(pt_path)
    if not os.path.exists(train):
        create_pt_pan2018(train_csv, train, root='', server_type=server)
    if not os.path.exists(val):
        create_pt_pan2018(val_csv, val, root='', server_type=server)

    # train the model
    train_model(train, val, model_name)

