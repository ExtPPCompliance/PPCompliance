# -*- coding: utf-8 -*-


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processor import DataProcessor
from tqdm import tqdm

torch.manual_seed(123)

vocab_size = 5000
embedding_size = 100  
num_classes = 11   
sentence_max_len = 128 
hidden_size = 256

num_layers = 1  
num_directions = 2 
lr = 5e-4
batch_size = 64   
epochs = 100

log_saving_file_PATH ='log/log.txt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Bi-LSTM
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(BiLSTMModel, self).__init__()
        
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        
        
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)
    
    def forward(self, x):
 
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]
        

        batch_size = x.size(1)
 
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
  
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        

        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  
        
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        x = self.liner(x)
        x = self.act_func(x)
        return x
        
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        labels=labels.float()
        preds = model(datas)
        loss = loss_func(preds, labels)
        
        loss_val += loss.item() * datas.size(0)
        

        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc),file=open(log_saving_file_PATH,'a'))
    return test_acc

def train(model, train_loader,test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    loop_iter = tqdm(range(epochs))
    for epoch in loop_iter:
        loop_iter.desc = 'Epoch['+str(epoch)+']'
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            labels=labels.float()
            preds = model(datas)
            # print(preds, labels)
            loss = loss_func(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() 

            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        if(epoch % 2 == 0):
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc),file=open(log_saving_file_PATH,'a'))
            test_acc = test(model, test_loader, loss_func)
            # if(best_val_acc < test_acc):
            #     best_val_acc = test_acc
            #     best_model_params = copy.deepcopy(model.state_dict())
        if(epoch % 10 == 0):
            torch.save(model.state_dict(),'log/EPO:{}-ACC:{:.3}.pth'.format(epoch,test_acc))
    # model.load_state_dict(best_model_params)
    return model
if __name__=='__main__':
    processor = DataProcessor()
    train_datasets, test_datasets = processor.get_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
    print(model,file=open(log_saving_file_PATH,'a'))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    model = train(model, train_loader, test_loader, optimizer, loss_func, epochs)


