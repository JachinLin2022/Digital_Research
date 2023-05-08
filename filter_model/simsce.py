import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import heapq
import numpy as np
import random
import os
from torch import nn
# from torchsummary import summary

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


MODEL_PATH = '/home/linzhisheng/Digital_Research/SimCSE-main/result/my-sup-simcse-bert-base-uncased-test'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = 'cuda:3'
model = AutoModel.from_pretrained(MODEL_PATH).to(device)

    
def main(top_n=5):
    # Import our models. The package will take care of downloading the models automatically

    # Tokenize input texts

    # data = pd.read_csv(
    # '/home/linzhisheng/Digital_Research/filter_model/data/filter_model_result_roberta_large_uncased.csv',nrows=100)
    # data = pd.read_csv(
        # '/home/linzhisheng/Digital_Research/filter_model/data/label_tweets.csv')
    data = pd.read_csv(
        '/home/linzhisheng/Digital_Research/filter_model/data/filter_model_validate_set.csv')

    # data = pd.concat([data, data2], ignore_index=True)
    # data['relative'] = np.random.randint(2, size=len(data))

    texts = data["full_text"].tolist()

    # print(texts)
    inputs = tokenizer(texts, padding=True,
                       truncation=True, return_tensors="pt").to(device)

    # # Get the embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True,
                           return_dict=True).pooler_output.cpu()

    # cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)


    # print(cos_sim.shape)


    max_top_n = min(len(data[data['relative']==1]),len(data[data['relative']==0]))
    if top_n > max_top_n:
        return
    # # eval
    all = 0
    neg = 0
    pos = 0
    ttt = 0
    for i in range(len(data)):
        # print('\n')
        # if data.iloc[i]['relative'] == 1:
        #     continue
        all = all + 1
        pos_res = []
        neg_res = []

        for j in range(len(data)):
            if i == j:
                continue
            cosine_sim = 1 - cosine(embeddings[i], embeddings[j])
            
            # cosine_sim = cos_sim[i][j]
            
            if data.iloc[i]['relative'] == data.iloc[j]['relative']:
                
                
                pos_res.append((i, j, cosine_sim, data.iloc[j]['relative']))
            else:
                
                neg_res.append((i, j, cosine_sim, data.iloc[j]['relative']))
        # print(heapq.nlargest(3, pos_res, key=lambda x: x[2]))
        pos_top_n = heapq.nlargest(top_n, pos_res, key=lambda x: x[2])
        neg_top_n = heapq.nlargest(top_n, neg_res, key=lambda x: x[2])
        # print((len(pos_top_n), len(neg_top_n)))
        if sum(x[2] for x in pos_top_n)/len(pos_top_n) >= sum(x[2] for x in neg_top_n)/len(neg_top_n):
            pos = pos + 1
        if sum(x[2] for x in pos_top_n)/len(pos_top_n) < sum(x[2] for x in neg_top_n)/len(neg_top_n):
            neg = neg + 1

        # print((sum(x[2] for x in pos_res)/len(pos_res), sum(x[2] for x in neg_res)/len(neg_res)))
        ttt = ttt + (1 if sum(x[2] for x in pos_res)/len(pos_res) >= sum(x[2] for x in neg_res)/len(neg_res) else 0)
        # break
    print((top_n, len(data), pos, neg,pos/len(data)))
    # print(ttt,all)

if __name__ == '__main__':
    for i in range(50):
        main(i+1)
