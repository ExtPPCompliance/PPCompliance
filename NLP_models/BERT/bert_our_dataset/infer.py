from train import CustomEffNet,LitPrivacy
from transformers import BertTokenizer, BertModel,BertConfig
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomEffNet()
model.load_state_dict(torch.load('./logs/bert-base-uncased/version_1/checkpoints/epoch=39-valid_loss=0.6967-valid_acc=0.8082.ckpt')['state_dict'])
model.to(device)
model.eval()
embeddingmodel = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sm = torch.nn.Softmax()


sentence = 'This site provides any third party cookies and makes no effort to track you.'

a = tokenizer.encode(sentence, add_special_tokens=True)
embedding_res = embeddingmodel(torch.tensor(a).unsqueeze(0))[1].detach().to(device)

with torch.no_grad():
    pred = model(embedding_res).squeeze()
pred = sm(pred)

pred = pred.detach().cpu().tolist()
print(pred)