# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class CFG:
    seed = 42
    n_fold=4
    base_dir=''
    filename='zimmeck_dataset_filtered.csv'
    infer_filename='policy_all_in_one_filter_purged.csv'
    # infer_filename='liushuang_dataset.csv'

torch.manual_seed(123)
class DataProcessor(object):
    def read_text(self):

        datas = []
        labels = []
        # read data
        df_all = pd.read_csv(os.path.join(CFG.base_dir,CFG.filename))
        df_all.dropna(inplace=True)

        # k-fold
        skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

        for train_idx, valid_idx in skf.split(df_all['review'], df_all["label"]):
            df_train = df_all.iloc[train_idx]
            df_valid = df_all.iloc[valid_idx]
            # break
        
        # train data set
        train_datas = df_train['review'].values
        train_labels = df_train['label'].values
        # test data set
        test_datas = df_valid['review'].values
        test_labels = df_valid['label'].values
        
        return train_datas, train_labels, test_datas,test_labels
    
    def read_infer_text(self):

        # read data
        df_all = pd.read_csv(os.path.join(CFG.base_dir,CFG.infer_filename))
        df_all.dropna(inplace=True)

        # test data set
        test_datas = df_all['review'].values
        test_labels = df_all['label'].values

        return test_datas,test_labels

    def word_count(self, datas):
        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                word = word.lower() 
                if(word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item:item[1], reverse=True)
        return  word_count_sorted
    
    def word_index(self, datas, vocab_size):
        word_count_sorted = self.word_count(datas)
        word2index = {}
        word2index["<unk>"] = 0
        word2index["<pad>"] = 1

        vocab_size = min(len(word_count_sorted), vocab_size)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + 2
          
        return word2index, vocab_size
    
    def get_datasets(self, vocab_size, embedding_size, max_len):

        train_datas, train_labels,test_datas, test_labels = self.read_text()
        word2index, vocab_size = self.word_index(train_datas, vocab_size)
        
        train_features = []
        for data in train_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower() 
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"]) 
                if(len(feature)==max_len): 
                    break

            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            train_features.append(feature)
            
        test_features = []
        for data in test_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower() 
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])
                if(len(feature)==max_len): 
                    break
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            test_features.append(feature)
            
        train_features = torch.LongTensor(train_features)
        train_labels = torch.LongTensor(train_labels)
        train_labels = nn.functional.one_hot(train_labels,num_classes=-1)
        test_features = torch.LongTensor(test_features)
        test_labels = torch.LongTensor(test_labels)
        test_labels= nn.functional.one_hot(test_labels,num_classes=-1)

        embed = nn.Embedding(vocab_size + 2, embedding_size)
        train_features = embed(train_features)
        test_features = embed(test_features)

        train_features = Variable(train_features, requires_grad=False)
        train_datasets = torch.utils.data.TensorDataset(train_features, train_labels)
        
        test_features = Variable(test_features, requires_grad=False)
        test_datasets = torch.utils.data.TensorDataset(test_features, test_labels)
        return train_datasets, test_datasets

    def get_infer_datasets(self, vocab_size, embedding_size, max_len):

        test_datas, test_labels = self.read_infer_text()
        word2index, vocab_size = self.word_index(test_datas, vocab_size)
                    
        test_features = []
        for data in test_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower() 
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"]) 
                if(len(feature)==max_len): 
                    break

            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            test_features.append(feature)
            
        test_features = torch.LongTensor(test_features)
        test_labels = torch.LongTensor(test_labels)
        test_labels= nn.functional.one_hot(test_labels,num_classes=-1)

        embed = nn.Embedding(vocab_size + 2, embedding_size)
        test_features = embed(test_features)
        
        test_features = Variable(test_features, requires_grad=False)
        test_datasets = torch.utils.data.TensorDataset(test_features, test_labels)
        return test_datasets
    