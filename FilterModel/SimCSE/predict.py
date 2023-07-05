from transformers import AutoModel, AutoTokenizer,AutoConfig
from torch.utils.data import DataLoader
import torch
import pandas as pd
import json
from datasets import *
from simcse.models import RobertaForCL, BertForCL
from simcse.trainers import CLTrainer
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import f1_score,precision_score, recall_score, accuracy_score
import numpy as np


MODEL_PATH = '/home/linzhisheng/Digital_Research/SimCSE/result/bert-base-uncased-p-0'
DEVICE = torch.device('cuda:3')
MAX_LENGTH = 32


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
    eval_data = load_dataset('csv', data_files='/home/linzhisheng/Digital_Research/filter_model/data/filter_model_manual_400.csv',split='train')
    train_data = train_data.select(range(88))
    train_labels = train_data["label"]
    train_texts = train_data["full_text"]
    train_inputs = tokenizer(train_texts, padding=True,max_length = MAX_LENGTH,
                        truncation=True, return_tensors="pt").to(DEVICE)




    texts = eval_data["full_text"]
    labels = eval_data["label"]
    inputs = tokenizer(texts, padding=True,
                        truncation=True, return_tensors="pt").to(DEVICE)
    
    # eval_data = eval_data.map(
    #     process_date,
    #     batched=True,
    #     remove_columns=eval_data.column_names
    #     )
    # print(eval_data)
    # features = []
    # for i in range(len(eval_data)):
    #     sample = eval_data[i]
    #     features.append({
    #         # 'label': sample['label'],
    #         'input_ids': sample['input_ids'][0],
    #         # 'token_type_ids': sample['token_type_ids'][0],
    #         'attention_mask': sample['attention_mask'][0]
    #     })
                    
    # batch = tokenizer.pad(
    #             features,
    #             padding=True,
    #             return_tensors="pt",
    #         ).to(DEVICE)
    #     # for i in batch['input_ids']:
    #     #     print(len(i))
    # batch = {k: batch[k].view(len(eval_data), -1) for k in batch}
    
    model.eval()

    with torch.no_grad():
        base_embeddings = model(**train_inputs, output_hidden_states=True,
                            return_dict=True).last_hidden_state[:, 0].cpu()
        embeddings = model(**inputs, output_hidden_states=True,
                            return_dict=True).last_hidden_state[:, 0].cpu()
    print(base_embeddings.shape)
    print(embeddings.shape)
    cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), base_embeddings.unsqueeze(0), dim=-1)
    # 相似度匹配
    # target = pd.read_csv('/home/linzhisheng/Digital_Research/filter_model/data/filter_model_manual_400.csv')
    
    # # types = ['random_sample','max_same_sample','max_diff_sample','min_same_sample','min_diff_sample']
    
    
    # # target['random_sample'] = ''
    # # target['random_sample_sim'] = ''
    # # target['max_same_sample'] = ''
    # # target['max_same_sample_sim'] = ''
    # # target['max_diff_sample'] = ''
    # # target['max_diff_sample_sim'] = ''

    # # target['min_same_sample'] = ''
    # # target['min_same_sample_sim'] = ''
    # # target['min_diff_sample'] = ''
    # # target['min_diff_sample_sim'] = ''
    

    
    
    # topk_sim_scores, topk_indices = torch.topk(cos_sim, len(train_texts), dim=-1)  # 获取相似度最高的10个base_embeddings
    # _, bottomk_indices = torch.topk(cos_sim, len(train_texts), dim=-1, largest=False)  # 获取相似度最低的10个base_embeddings的下标

    # for num in [1,5,10,20,30]:
    
    # # num = 10
    #     # random
    #     import random
    #     for i in range(len(texts)):
    #         sample_text = ''
    #         # sim = 0
    #         for j in range(num):
    #             random_idx = random.randint(0, len(train_labels)-1)
    #             # sim = sim + cos_sim[i][random_idx]
    #             sample_text = sample_text + f'Sample {j+1}:[' + train_texts[random_idx] + f']\nSample {j+1} label:[{train_labels[random_idx]}]\n'
                
    #         target.at[i,f'random_sample_{num}'] = sample_text
    #         # target.at[i,'random_sample_sim'] = sim

    #     # max_same
    #     for i in range(len(texts)):
    #         sample_text = ''
    #         # sim = 0
    #         count = 0
    #         for j in range(len(train_texts)):
    #             if count == num:
    #                 break
    #             if labels[i] == train_labels[j]:
    #                 count = count + 1
    #                 # sim = cos_sim[i][j]
    #                 sample_text = sample_text + f'Sample {count}:[' + train_texts[topk_indices[i][j]] + f']\nSample {j+1} label:[{train_labels[topk_indices[i][j]]}]\n'
    #                 # sample_text = sample_text + f'Sample {count}:[' + train_texts[topk_indices[i][j]] + f'], sim:[{cos_sim[i][topk_indices[i][j]]}]\n'
    #         target.at[i,f'max_same_sample_{num}'] = sample_text
    #         # target.at[i,f'max_same_sample_sim'] = sim

    #     # max_diff
    #     for i in range(len(texts)):
    #         sample_text = ''
    #         # sim = 0
    #         count = 0
    #         for j in range(len(train_texts)):
    #             if count == num:
    #                 break
    #             if labels[i] != train_labels[j]:
    #                 count = count + 1
    #                 # sim = cos_sim[i][j]
    #                 sample_text = sample_text + f'Sample {count}:[' + train_texts[topk_indices[i][j]] + f']\nSample {j+1} label:[{train_labels[topk_indices[i][j]]}]\n'
    #                 # sample_text = sample_text + f'Sample {count}:[' + train_texts[topk_indices[i][j]] + f'], sim:[{cos_sim[i][topk_indices[i][j]]}]\n'
    #         target.at[i,f'max_diff_sample_{num}'] = sample_text
    #         # target.at[i,f'max_same_sample_sim'] = sim

    #     # min_same
    #     for i in range(len(texts)):
    #         sample_text = ''
    #         # sim = 0
    #         count = 0
    #         for j in range(len(train_texts)):
    #             if count == num:
    #                 break
    #             if labels[i] == train_labels[j]:
    #                 count = count + 1
    #                 # sim = cos_sim[i][j]
    #                 sample_text = sample_text + f'Sample {count}:[' + train_texts[bottomk_indices[i][j]] + f']\nSample {j+1} label:[{train_labels[bottomk_indices[i][j]]}]\n'
    #                 # sample_text = sample_text + f'Sample {count}:[' + train_texts[bottomk_indices[i][j]] + f'], sim:[{cos_sim[i][bottomk_indices[i][j]]}]\n'
    #         target.at[i,f'min_same_sample_{num}'] = sample_text
    #         # target.at[i,f'max_same_sample_sim'] = sim
            
            
    #     # min_diff
    #     for i in range(len(texts)):
    #         sample_text = ''
    #         # sim = 0
    #         count = 0
    #         for j in range(len(train_texts)):
    #             if count == num:
    #                 break
    #             if labels[i] != train_labels[j]:
    #                 count = count + 1
    #                 # sim = cos_sim[i][j]
    #                 sample_text = sample_text + f'Sample {count}:[' + train_texts[bottomk_indices[i][j]] + f']\nSample {j+1} label:[{train_labels[bottomk_indices[i][j]]}]\n'
    #                 # sample_text = sample_text + f'Sample {count}:[' + train_texts[topk_indices[i][j]] + f'], sim:[{cos_sim[i][topk_indices[i][j]]}]\n'
    #         target.at[i,f'min_diff_sample_{num}'] = sample_text
    #         # target.at[i,f'max_same_sample_sim'] = sim
    # print(target)
    # target.to_csv('/home/linzhisheng/Digital_Research/filter_model/data/hard_negative_test_with_label.csv',index=False)
    
    # 分类 
    # pos_idx = [idx for idx, label in enumerate(train_labels) if label == 1]
    # neg_idx = [idx for idx, label in enumerate(train_labels) if label == 0]
    # pos_anchor = torch.mean(base_embeddings[pos_idx],dim=0,keepdim=True)
    # neg_anchor = torch.mean(base_embeddings[neg_idx],dim=0,keepdim=True)
    # print('anchor shape',pos_anchor.shape)
    # pos_cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), pos_anchor.unsqueeze(0), dim=-1)
    # neg_cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), neg_anchor.unsqueeze(0), dim=-1)
    # right = 0
    # wrong = 0
    # target['cl_label'] = ''
    # preds = []
    # for i in range(len(texts)):
    #     if pos_cos_sim[i][0] >= neg_cos_sim[i][0]:
    #         preds.append(1)
    #         target.at[i,'cl_label'] = 1
    #         if labels[i] == 1:
    #             right = right + 1
    #         else:
    #             wrong = wrong + 1
    #     else:
    #         preds.append(0)
    #         target.at[i,'cl_label'] = 0
    #         if labels[i] == 0:
    #             right = right + 1
    #         else:
    #             wrong = wrong + 1
                
    
    # acc = right/(right+wrong)
    # print(acc)
    
    # preds_np = np.array(preds)
    # labels_np = np.array(labels)
    
    
    # f1 = f1_score(labels_np, preds_np)
    # precision = precision_score(labels_np, preds_np)
    # recall = recall_score(labels_np, preds_np)
    # accuracy = accuracy_score(labels_np, preds_np)

    # print(f"F1-score: {f1:.5f}")
    # print(f"Precision: {precision:.5f}")
    # print(f"Recall: {recall:.5f}")
    # print(f"Accuracy: {accuracy:.5f}")
    
    # f1 = f1_score(labels_np, preds_np)
    # print(f"F1 score: {f1}")
    # target.to_csv('/home/linzhisheng/Digital_Research/filter_model/data/filter_model_manual_400.csv',index=False)
    
if __name__ == '__main__':
    main()