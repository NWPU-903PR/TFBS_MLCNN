# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from model_bio import bioinfor16, VNet
'''
data shape：（#samples，101,4,1）

'''
parser = argparse.ArgumentParser(description='PyTorch  Training')

parser.add_argument('--epochs', default=19, type=int,
                    help='number of epochs used in bioinfor16')
parser.add_argument('--batch-size', default=100, type=int,
                    help='mini-batch size used in bioinfor16')
parser.add_argument('--lr',default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--saveResPath', default='../res', type=str,
                    help='saveResPath')
parser.add_argument('--pathAndData', default='', type=str,
                    help='path+.npy')
parser.add_argument('--gpu', default=0, type=int,
                    help='idx of gpu')
parser.add_argument('--vnode', default=100, type=int, help='')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--corruptedRate', default=0.0, type=float, help=' ')
parser.add_argument('--posmetanum', default=500, type=int, help=' ')

parser.set_defaults(augment=True)

global args
args = parser.parse_args()
print()
print(args)
torch.cuda.set_device(int(args.gpu))
# parameters
SEQUENCE_WIDTH = 4
identity = args.pathAndData.split('.')[0].split('/')[-1]
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
def eval_clf(y_pred,y_test): 

    f1 = []
    auc = []
    aupr = []

    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    aupr = metrics.average_precision_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    overall_accuracy = metrics.accuracy_score(y_test, y_pred)     
    return f1,auc,aupr,overall_accuracy,tn, fp, fn, tp 

saveModelPath = args.saveResPath + '/meta/model'
mkdir(args.saveResPath)

# init the net pramaters
def weight_init(m):
    # for m in self.modules():
    '''
    usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)  # nn.init.constant(param,0.0)

def build_model():
    model = bioinfor16()
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def batch_generator(batch_size, data, labels, corrupted_labels, data_index):
    size = data.shape[0]
    idx_array = np.arange(size)
    np.random.shuffle(idx_array)
    n_batch = int(np.ceil(size / float(batch_size)))
    batches = [(int(i * batch_size), int(min(size, (i + 1) * batch_size))) for i in range(0, n_batch)]
    for batch_index, (start, end) in enumerate(batches):
        batch_ids = idx_array[start:end]
        if corrupted_labels is not None:
            yield torch.from_numpy(data[batch_ids]).type(torch.FloatTensor), torch.from_numpy(labels[batch_ids]).type(
                torch.LongTensor), torch.from_numpy(corrupted_labels[batch_ids]).type(torch.LongTensor), torch.from_numpy(data_index[batch_ids]).type(torch.LongTensor)
        else:
            yield torch.from_numpy(data[batch_ids]).type(torch.FloatTensor), torch.from_numpy(labels[batch_ids]).type(torch.LongTensor)


def corrupted_label(target,corruptedRate):
    random.seed(1)
    sel_index = random.sample(range(len(target)), int(len(target) * corruptedRate))
    target[sel_index] = ~target[sel_index] + 2
    return target

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.1 ** int(epoch >= 10)) * (0.1 ** int(epoch >= 14)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def oneHot_data_encode(sequences):
    return (np.arange(1, SEQUENCE_WIDTH + 1) == sequences.flatten()[:, None]).astype(np.float32).reshape(len(sequences), sequences.shape[1], SEQUENCE_WIDTH)[..., None]

def split_data_meta_train_test(pathAndData,posmetanum):
    data = np.load(pathAndData) # pathAndData is path and .npy data
    data = oneHot_data_encode(data)# 
    data = data.transpose((0, 3, 1, 2))     

    pathAndLabel = pathAndData.split('.')[0]+'.narrowPeak.targets.npy'
    label = np.load(pathAndLabel)
    num_pos = data.shape[0]//2
    assert (label[num_pos-1] == 1) and (label[num_pos] == 0)
    meta_pos = data[0:2*posmetanum:2]
    test_pos = data[1:2*posmetanum:2]
    train_pos = data[2*posmetanum:num_pos]
    meta_neg = data[num_pos:num_pos+2*posmetanum:2]
    test_neg = data[num_pos+1:num_pos+2*posmetanum:2]
    train_neg = data[num_pos+2*posmetanum:]
    #concatenate
    meta_data = np.concatenate((meta_pos, meta_neg), axis=0)
    test_data = np.concatenate((test_pos, test_neg), axis=0)
    train_data = np.concatenate((train_pos, train_neg), axis=0)
    #label
    meta_label = np.concatenate((np.array([1] * posmetanum), np.array([0] * posmetanum)), axis=0)
    test_label = np.concatenate((np.array([1] * posmetanum), np.array([0] * posmetanum)), axis=0)
    train_label = np.concatenate((np.array([1] * (num_pos-2*posmetanum)), np.array([0] * (num_pos-2*posmetanum))), axis=0)
    # index
    train_index = np.array(np.arange(train_data.shape[0]))
    return meta_data, meta_label, test_data, test_label, train_data, train_label, train_index

def train_nometa_mixdata_model( test_data, test_label, train_data,train_index, train_label,cor_labels,n_batch):
    model = build_model()
    # init model params
    model.apply(weight_init)
    # optimizer_a = torch.optim.Adadelta(model.params(), lr=1.0, rho=0.9, eps=1e-08, weight_decay=0)
    optimizer_a = torch.optim.SGD(model.params(), args.lr, momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    cudnn.benchmark = True
    model_loss = []
    # record training information
    all_record = [['epoch', 'testF1', 'testAUC', 'testAUPR', 'testAcc', 'tn', 'fp', 'fn', 'tp']]
    for i in range(args.epochs):
        # create iter
        tra_loader = batch_generator(args.batch_size, train_data, train_label, cor_labels,train_index)
        model.train()
        adjust_learning_rate(optimizer_a, i)
        for j in range(n_batch):

            input, _, cor_target, _ = next(iter(tra_loader))
            input_var = to_var(input, requires_grad=False)
            target_var = to_var(cor_target, requires_grad=False)

            y_f = model(input_var)
            cost_w = F.cross_entropy(y_f, target_var)

            optimizer_a.zero_grad()
            cost_w.backward()
            optimizer_a.step()

            net_l = cost_w.item()
            model_loss.append(net_l / (1 +j))
        # evaluate each epoch
        model.eval()
        tes_loader = batch_generator(args.batch_size, test_data, test_label, None, None)
        all_test_y = np.array([])
        all_pre = np.array([])
        for (test_x, test_y) in tes_loader:
            test_x = to_var(test_x, requires_grad=False)
            test_y = test_y.numpy()
            with torch.no_grad():
                pre_label = model(test_x)
                pre_label = torch.max(pre_label, 1)[1].cpu().data.numpy()
                all_test_y = np.r_[all_test_y, test_y]
                all_pre = np.r_[all_pre, pre_label]
        f1, auc, aupr, overall_accuracy, tn, fp, fn, tp = eval_clf(all_pre, all_test_y)
        print('Epoch: %d/%d  '
              'Pre_test: F1:%.3f  |'
              'AUC:%.3f  |'
              'Acc:%.3f\t' % (i + 1, args.epochs, f1, auc, overall_accuracy))

        all_record.append([i + 1, f1, auc, aupr, overall_accuracy, tn, fp, fn, tp])
    # end training
    torch.save(model,args.saveResPath + '/nometa/model/' + identity + '.model.pkl')

'''
meta model
'''
def main():
    # create model
    model = build_model()
    # init model params
    model.apply(weight_init)
    # optimizer_a = torch.optim.Adadelta(model.params(), lr=1.0, rho=0.9, eps=1e-08, weight_decay=0)
    optimizer_a = torch.optim.SGD(model.params(), args.lr, momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    vnet = VNet(1, args.vnode, 1).cuda()
    # init model params
    vnet.apply(weight_init)
    # optimizer_c = torch.optim.SGD(vnet.params(), 1e-3, momentum=args.momentum, nesterov=args.nesterov,
    #                                 weight_decay=args.weight_decay)
    optimizer_c = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)
    cudnn.benchmark = True

    model_loss = []
    meta_model_loss = []

    # load data
    meta_data, meta_label, test_data, test_label, train_data, train_label,train_index = split_data_meta_train_test(args.pathAndData,args.posmetanum)
    # add in main
    labels_copy = train_label.copy()
    cor_labels = corrupted_label(labels_copy, args.corruptedRate)
    # num batches
    n_batch = int(np.ceil(train_data.shape[0] / float(args.batch_size)))
    # record training information
    all_record = [['epoch', 'testF1', 'testAUC', 'testAUPR', 'testAcc', 'tn', 'fp', 'fn', 'tp']]
    for i in range(args.epochs):
        # create iter
        tra_loader = batch_generator(args.batch_size, train_data, train_label, cor_labels,train_index)
        met_loader = batch_generator(args.batch_size, meta_data, meta_label, None,None)
        model.train()
        adjust_learning_rate(optimizer_a,i)
        #init record 
        lossWeight_loss = 0
        lossWeight_weight = 0 
        raw_label = 0
        record_c_label = 0
        train_data_index = 0
        for j in range(n_batch):
            input, target, corrupted_target, train_batch_index = next(iter(tra_loader)) # target is corrupted labels
            input_var = to_var(input, requires_grad=False)
            target_var = to_var(corrupted_target, requires_grad=False)

            meta_model = build_model()

            meta_model.load_state_dict(model.state_dict())
            y_f_hat = meta_model(input_var)
            cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))

            v_lambda = vnet(cost_v.data)
            if j == 0:
                lossWeight_loss = cost_v.data.cpu().numpy()
                lossWeight_weight = v_lambda.data.cpu().numpy()
                raw_label = target.data.cpu().numpy()[:, np.newaxis]#
                record_c_label = corrupted_target.data.cpu().numpy()[:, np.newaxis]
                train_data_index = train_batch_index.data.cpu().numpy()[:, np.newaxis]#
            else:
                lossWeight_loss = np.vstack((lossWeight_loss, cost_v.data.cpu().numpy()))#, axis=0)
                lossWeight_weight = np.vstack((lossWeight_weight, v_lambda.data.cpu().numpy()))#,axis=0)
                raw_label = np.vstack((raw_label, target.data.cpu().numpy()[:, np.newaxis]))#, axis=0)
                record_c_label = np.vstack((record_c_label, corrupted_target.data.cpu().numpy()[:, np.newaxis]))#,axis=0)
                train_data_index = np.vstack((train_data_index, train_batch_index.data.cpu().numpy()[:, np.newaxis]))#,

            l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            meta_lr = args.lr * ((0.1 ** int(i >= 18)) * (0.1 ** int(i >= 28))) 
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            try:
                met_, met_target = next(iter(met_loader))
            except:
                met_loader = batch_generator(args.batch_size, meta_data, meta_label, None,None)
                met_, met_target = next(iter(met_loader))
            met_var = to_var(met_, requires_grad=False)
            met_target_var = to_var(met_target.type(torch.LongTensor), requires_grad=False)

            y_g_hat = meta_model(met_var)
            l_g_meta = F.cross_entropy(y_g_hat, met_target_var)

            optimizer_c.zero_grad()
            l_g_meta.backward()
            optimizer_c.step()

            y_f = model(input_var)
            cost_w = F.cross_entropy(y_f, target_var, reduce=False)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))

            with torch.no_grad():
                w_new = vnet(cost_v)


            l_f = torch.sum(cost_v * w_new)/len(cost_v)

            optimizer_a.zero_grad()
            l_f.backward()
            optimizer_a.step()

            meta_l = l_g_meta.item()   # each mini-batch has a value
            meta_model_loss.append(meta_l / (1 + j))

            net_l = l_f.item()  # each mini-batch has a value
            model_loss.append(net_l / (1 +j))
        weightLossCorrTrueLabel = np.concatenate((lossWeight_weight, lossWeight_loss, record_c_label, raw_label, train_data_index), axis=1)
        # evaluate each epoch
        model.eval()
        tes_loader = batch_generator(args.batch_size, test_data, test_label, None,None)
        all_test_y = np.array([])
        all_pre = np.array([])
        for (test_x, test_y) in tes_loader:
            test_x = to_var(test_x, requires_grad=False)
            test_y = test_y.numpy()
            with torch.no_grad():
                pre_label = model(test_x)
                pre_label = torch.max(pre_label, 1)[1].cpu().data.numpy()
                all_test_y = np.r_[all_test_y, test_y]
                all_pre = np.r_[all_pre, pre_label]
        f1, auc, aupr, overall_accuracy, tn, fp, fn, tp = eval_clf(all_pre, all_test_y)
        # np.savetxt(args.saveResPath +'/' + identity+'.pre.epoch.'+str(i+1)+'.txt', all_pre)
        # np.savetxt(args.saveResPath +'/' + identity + '.true.epoch.' + str(i+1) +'.txt', all_test_y)
        print('Epoch: %d/%d  '
              'Pre_test: F1:%.3f  |'
              'AUC:%.3f  |'
              'Acc:%.3f\t' % (i + 1, args.epochs, f1, auc, overall_accuracy))

        all_record.append([i + 1, f1, auc, aupr, overall_accuracy, tn, fp, fn, tp])
  
    torch.save(model, saveModelPath + '/' + identity + '.model.pkl')
    torch.save(vnet,  saveModelPath + '/' + identity+ '.vnet.pkl')
    # train no meta model
    train_nometa_mixdata_model( test_data, test_label, train_data,train_index, train_label,cor_labels,n_batch)























if __name__ == '__main__':
    main()
