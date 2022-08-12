from train import CustomEffNet,LitPrivacy
from transformers import BertTokenizer, BertModel,BertConfig
import torch
import os
from bs4 import BeautifulSoup
import csv

#split article into sentences
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    res_sentences=[]
    for sentence in sentences:
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.!#$%^&*()_+-=[]{}?/:;\'\"')
        sentence = ''.join(filter(whitelist.__contains__, sentence))
        if len(sentence)>500:
            sentence=sentence[:500]
        res_sentences.append(sentence)
    #print(sentences)
    return res_sentences


# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomEffNet()
model.load_state_dict(torch.load('./logs/bert-base-uncased/version_1/checkpoints/epoch=39-valid_loss=0.6967-valid_acc=0.8082.ckpt')['state_dict'])
model.to(device)
model.eval()
embeddingmodel = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sm = torch.nn.Softmax()

# load sentences
policy_folder='../../raw_data/privacy'
policy_conclude_predict=[]
policy_raw_result=[]
for root, dirs, files in os.walk(policy_folder):
    from tqdm import tqdm
    count=0
    for file in tqdm(files):
        # count+=1
        # if count>20:
        #     break
        
        ext_id=file[:-5]
        ext_result=[0]*21
        # print(ext_result)
        with open(os.path.join(root, file), 'r') as f:
            try:
                tmp = f.read()
            except:
                continue
            soup = BeautifulSoup(tmp, 'html.parser')
            text=soup.get_text()
            text=text.replace('  ', '')
            text=text.replace('\\n', '')
            text=text.replace('\\t', '')
            text=text.replace('\\', '')
            text=text.replace('xe2x80x9c', '')
            text=text.replace('xe2x80x9d', '')
            sentences=split_into_sentences(text)
            for sentence in sentences:
                a = tokenizer.encode(sentence, add_special_tokens=True)
                embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

                with torch.no_grad():
                    pred = model(embedding_res).squeeze()
                pred = sm(pred)
                pred = pred.detach().cpu().tolist()
                # pred_res=pred.index(max(pred))
                pred_res=sorted(pred)
                pred_res1=pred.index(pred_res[-1])
                pred_res2=pred.index(pred_res[-2])
                ext_result[pred_res1]+=1
                ext_result[pred_res2]+=1
                with open('./privacy_conclude_raw.csv','a') as f:
                    csvf=csv.writer(f)
                    res=[ext_id,sentence]+[str(i) for i in pred]
                    csvf.writerow(res)
                # print(ext_id,pred_res)
        with open('./privacy_conclude_result2.csv','a') as f:
            csvf=csv.writer(f)
            csvf.writerow([ext_id]+[str(i) for i in ext_result])
        policy_conclude_predict.append([ext_id]+ext_result)

with open('./privacy_conclude_result_last.csv','w') as f:
    csvf=csv.writer(f)
    csvf.writerows(policy_conclude_predict)
'''        
sentence = 'This site provides any third party cookies and makes no effort to track you.'

a = tokenizer.encode(sentence, add_special_tokens=True)
embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

with torch.no_grad():
    pred = model(embedding_res).squeeze()
pred = sm(pred)

pred = pred.detach().cpu().tolist()
print
'''