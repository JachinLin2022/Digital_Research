from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import pandas as pd

df = pd.read_csv('/home/linzhisheng/Digital_Research/data/filter_tweets_sample.csv')
dataset = [] 
for i in range(len(df)): 
    example = InputExample(
        guid=i, 
        text_a=df.iloc[i]['full_text'],
        label=int(df.iloc[i]['relative'])
        ) 
    dataset.append(example)

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "cardiffnlp/twitter-roberta-base-sentiment-latest")
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
        "negative": ["false", "wrong", "no"],
        "positive": ["true", "right", "yes"],
    },
    tokenizer = tokenizer,
)

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)
use_cuda = True
if use_cuda:
    promptModel=  promptModel.cuda()

data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size= 16
)

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

for epoch in range(10):
    tot_loss = 0
    for step, inputs in enumerate(data_loader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
torch.save(promptModel.state_dict(), 'twitter-roberta-base-sentiment-latest.pth')
