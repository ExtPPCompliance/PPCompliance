from train import CustomEffNet,LitPrivacy
from transformers import BertTokenizer, BertModel,BertConfig
import torch
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv
import pandas as pd

def match_after_label_mapping(pred,target):
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


# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomEffNet()
model.load_state_dict(torch.load('./logs/bert-base-uncased/version_0/checkpoints/last.ckpt')['state_dict'])
model.to(device)
model.eval()
embeddingmodel = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sm = torch.nn.Softmax()

# load sentences
policy_file='./policy_all_in_one_filter_purged.csv'
col_list=['label','review','filename']
df=pd.read_csv(policy_file, usecols=col_list)

sentences=df['review']
target=df['label']
correct=0
for sentence in tqdm(sentences):
    a = tokenizer.encode(sentence, add_special_tokens=True)
    embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

    with torch.no_grad():
        pred = model(embedding_res).squeeze()
    pred = sm(pred)
    pred = pred.detach().cpu().tolist()
    # pred_res=pred.index(max(pred))
    pred_res=sorted(pred)
    pred_res1=pred.index(pred_res[-1])
    target_res=target[df.index[df['review'] == sentence].tolist()[0]]
    if match_after_label_mapping(pred_res1,target_res):
        correct += 1
    # print(pred_res1,target_res)
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(sentences), 100. * correct / len(sentences)) )
