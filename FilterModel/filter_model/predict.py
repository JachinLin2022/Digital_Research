from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import pandas as pd
import json
from sklearn.metrics import f1_score,precision_score, recall_score, accuracy_score
import numpy as np
def main():
    # read config
    with open('config.json', 'r') as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))
    epoch = config['epoch']
    lr = config['lr']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    model_path = config['model_path']
    state_dict_path = config['local_state_dict_path']
    device = config['device']
    train_data_path = config['train_data_path']
    eval_data_path = config['eval_data_path']
    eval_result_path = config['eval_result_path']
    validate_data_path = config['validate_data_path']
    
    df = pd.read_csv(eval_data_path)
    dataset = [] 
    for i in range(len(df)): 
        example = InputExample(
            guid=i, 
            text_a=str(df.iloc[i]['full_text'])
            ) 
        dataset.append(example)
        
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", model_path)


    classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
        "negative",
        "positive"
    ]

    promptTemplate = ManualTemplate(
        # text = '{"placeholder":"text_a"} It was {"mask"} that the above text is related to the price trend of Bitcoin.',
        text = '{"placeholder":"text_a"} Is it related to the price trend of Bitcoin? {"mask"}.',
        tokenizer = tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "negative": ["false", "wrong", "no", "False", "Wrong", "No"],
            "positive": ["true", "right", "yes", "True", "Right", "Yes"],
            # "negative": ["false", "wrong"],
            # "positive": ["true", "right"],
        },
        tokenizer = tokenizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size= eval_batch_size
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )
    promptModel.load_state_dict(torch.load(state_dict_path))

    use_cuda = True
    to_device = torch.device(device)
    if use_cuda:
        promptModel = promptModel.cuda(to_device)

    promptModel.eval()
    predict_result = []
    with torch.no_grad():
        count = 0
        for batch in data_loader:
            if use_cuda:
                batch = batch.cuda(to_device)
            logits = promptModel(batch)

            preds = torch.argmax(logits, dim = -1)

            predict_result.extend(preds.tolist())
            print(len(predict_result)/len(df))
            # print(classes[preds])
    # print(predict_result)

    
    df['prompt_label'] = ''
    for index, row in df.iterrows():
        df.at[index, 'prompt_label'] = int(predict_result[index])
    df.to_csv(eval_result_path,index=False)
    
if __name__ == '__main__':
    main()