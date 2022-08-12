import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processor import DataProcessor
from bilstm_attention import BiLSTMModel
from tqdm import tqdm

torch.manual_seed(123) 

vocab_size = 5000  
embedding_size = 100   
num_classes = 54    
sentence_max_len = 128 
hidden_size = 256

num_layers = 1  
num_directions = 2  
lr = 5e-4
batch_size = 64   
epochs = 100

log_saving_file_PATH ='log/log.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def match_after_label_mapping(pred,target):
    # uncommen if test is other set
    # pred,target=target,pred
    
    # from zimmeck to our corpus
    if pred==0 and target==0:
        return True
    if pred!=0 and target==1:
        return True
    if pred>=1 and pred<=16 and target==14:
        return True
    if pred>=17 and target<=22 and target==1:
        return True
    if pred>=23 and pred<=46 and target==13:
        return True
    if pred>=47 and pred<=58 and target==15:
        return True
    return False

    # from liushaung to our corpus
    # if pred==target:
    #     return True
    # if pred==1 and target>=11 and target<=19:
    #     return True
    # if pred==0 and target==20:
    #     return True
    # return False

if __name__=='__main__':
    processor = DataProcessor()
    test_datasets = processor.get_infer_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
    model.load_state_dict(torch.load('./log_v2/EPO:90-ACC:0.801.pth'))
    model.to(device)
    model.eval()
    loss_func = nn.BCELoss()

    corrects = 0.0
    res_list=[]
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        labels=labels.float()
        preds = model(datas)

        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        for i in range(len(preds)):
            pred=preds[i]
            label=labels[i]
            corrects += match_after_label_mapping(pred,label)
            res_list.append(pred)

    test_acc = corrects / len(test_loader.dataset)
    print("Test Acc: {}".format( test_acc))
    # print("Test Acc: {}".format(test_acc),file=open(log_saving_file_PATH,'a'))

    print(test_acc)