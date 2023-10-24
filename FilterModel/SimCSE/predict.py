
# 对比学习过滤post
from transformers import AutoModel, AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader
import torch
import pandas as pd
import json
from datasets import *
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import f1_score,precision_score, recall_score, accuracy_score
import numpy as np


MODEL_PATH = '/home/linzhisheng/Digital_Research/SimCSE/result/bert-base-uncased-p-0'
DEVICE = torch.device('cuda:0')
MAX_LENGTH = 32
BATCH_SIZE = 1

model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# print(model)

def process_date(example):
    
    sent_features = tokenizer(
        example['full_text'],
        padding=True,
    )
    return sent_features


def main(): 
    train_data = load_dataset('csv', data_files='/home/linzhisheng/Digital_Research/filter_model/data/filter_model_data.csv',split='train')
    eval_data = pd.read_csv('/home/linzhisheng/Digital_Research/tweets_raw_update_2500.csv')
        
    print(train_data)
    print(eval_data)
    
    train_data = train_data.select(range(88))
    train_labels = train_data["label"]
    train_texts = train_data["full_text"]
    train_inputs = tokenizer(train_texts, padding=True,max_length = MAX_LENGTH,
                        truncation=True, return_tensors="pt").to(DEVICE)

    eval_data["full_text"] = eval_data["full_text"].astype(str)
    # eval_data["id"] = eval_data["id"].astype(str)
    # eval_data["day"] = eval_data["day"].astype(str)
    # eval_data["created_at"] = eval_data["created_at"].astype(str)
    # eval_data['label'] = ''
    texts = eval_data["full_text"].to_list()
    print(len(texts))
    
    # ids = eval_data["id"].to_list()
    # period = eval_data["day"].to_list()
    # create = eval_data["created_at"].to_list()
    eval_data['label'] = 0
    batchs = [texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(int(len(texts)/BATCH_SIZE))]
    # id_batchs = [ids[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(int(len(texts)/BATCH_SIZE))]
    # period_batchs = [period[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(int(len(texts)/BATCH_SIZE))]
    # create_batchs = [create[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(int(len(texts)/BATCH_SIZE))]
    print(len(batchs))
    model.eval()
    with torch.no_grad():
        base_embeddings = model(**train_inputs, output_hidden_states=True,
                                return_dict=True).last_hidden_state[:, 0].cpu()
    pos_idx = [idx for idx, label in enumerate(train_labels) if label == 1]
    neg_idx = [idx for idx, label in enumerate(train_labels) if label == 0]
    pos_anchor = torch.mean(base_embeddings[pos_idx],dim=0,keepdim=True)
    neg_anchor = torch.mean(base_embeddings[neg_idx],dim=0,keepdim=True)
    print('anchor shape',pos_anchor.shape)
    
    res = []
    for id,batch in enumerate(batchs):
        print(id,len(batchs))
        inputs = tokenizer(batch, padding=True,
            truncation=True, return_tensors="pt").to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True,
                                return_dict=True).last_hidden_state[:, 0].cpu()
        pos_cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), pos_anchor.unsqueeze(0), dim=-1)
        neg_cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), neg_anchor.unsqueeze(0), dim=-1)
        for i in range(len(batch)):
            if pos_cos_sim[i][0] >= neg_cos_sim[i][0]:
                # print(id*BATCH_SIZE+i)
                eval_data.at[id*BATCH_SIZE+i,'label'] = 1
                # res.append([id_batchs[id][i],batch[i],period_batchs[id][i], create_batchs[id][i]])
        
        # print(len(res))
        # break

    # # 设置列名
    # columns = ["id", "full_text","date","created_at"]
    # # 创建DataFrame对象
    # df = pd.DataFrame(res, columns=columns)
    # df["label1"] = ''
    # df["label2"] = ''
    # print(df)
    # df.to_csv('tweets_sample_2020_2023_filter.csv',index=False)

    print(eval_data)
    eval_data[['id','label']].to_csv('tweet_raw_update_2500_filter.csv',index=False)


    
if __name__ == '__main__':
    main()