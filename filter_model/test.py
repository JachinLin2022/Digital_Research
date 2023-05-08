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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

def train():
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
    validate_data_path = config['validate_data_path']
    

    df = pd.read_csv(train_data_path)
    dataset = [] 
    for i in range(len(df)): 
        example = InputExample(
            guid=i, 
            text_a=df.iloc[i]['full_text'],
            label=int(df.iloc[i]['relative'])
            ) 
        dataset.append(example)

    validate_df = pd.read_csv(validate_data_path)
    validate_dataset = [] 
    for i in range(len(validate_df)): 
        example = InputExample(
            guid=i, 
            text_a=validate_df.iloc[i]['full_text'],
            label=int(validate_df.iloc[i]['relative'])
            ) 
        validate_dataset.append(example)



    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", model_path)
    classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
        "negative",
        "positive"
    ]

    promptTemplate = ManualTemplate(
        text = '{"placeholder":"text_a"} It is {"mask"} that the above text is related to the price trend of Bitcoin.',
        # text = '{"placeholder":"text_a"} Is it related to the price trend of Bitcoin? {"mask"}.',
        tokenizer = tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            # "negative": ["false", "wrong", "no", "False", "Wrong", "No"],
            # "positive": ["true", "right", "yes", "True", "Right", "Yes"],
            "negative": ["false", "wrong","False", "Wrong"],
            "positive": ["true", "right", "True", "Right"],
        },
        tokenizer = tokenizer,
    )

    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )

    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size= train_batch_size
    )
    
    validation_dataloader = PromptDataLoader(
        dataset = validate_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size= eval_batch_size
    )

    to_device = torch.device(device)
    
    loss_func = torch.nn.CrossEntropyLoss()
    # loss_func.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    
    
    
    promptModel = torch.nn.parallel.DataParallel(promptModel, device_ids=[2])
    promptModel = promptModel.cuda(to_device)
    

    
    # validate eval
    # promptModel.eval()
    # allpreds = []
    # alllabels = []
    # with torch.no_grad():
    #     for step, inputs in enumerate(validation_dataloader):
    #         if use_cuda:
    #             inputs = inputs.cuda(to_device)
    #         logits = promptModel(inputs)
    #         labels = inputs['label']
    #         alllabels.extend(labels.cpu().tolist())
    #         allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    # print(f"origin acc: {acc}")
    
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    for epoch in range(epoch):
        tot_loss = 0
        promptModel.train()
        train_step = 0
        for step, inputs in enumerate(data_loader):
            inputs = inputs.cuda(to_device)
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            train_step = step
            if step %100 ==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

        # validate eval
        # promptModel.eval()
        # allpreds = []
        # alllabels = []
        # with torch.no_grad():
        #     for step, inputs in enumerate(validation_dataloader):
        #         if use_cuda:
        #             inputs = inputs.cuda(to_device)
        #         logits = promptModel(inputs)
        #         labels = inputs['label']
        #         alllabels.extend(labels.cpu().tolist())
        #         allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        # acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        # print("Epoch {}, average loss: {}, acc: {}".format(epoch, tot_loss/(train_step + 1), acc), flush=True)

    
    # torch.save(promptModel.state_dict(), state_dict_path)

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus  # 每个GPU对应一个进程
    mp.spawn(train, args=(world_size,), nprocs=num_gpus, join=True)