from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
import pandas as pd


df = pd.read_csv('/home/linzhisheng/Digital_Research/data/Tweets.csv')
dataset = [] 
for i in range(len(df)): 
    example = InputExample(
        guid=i, 
        text_a=str(df.iloc[i]['full_text'])
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

data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size= 64
)

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)
promptModel.load_state_dict(torch.load('twitter-roberta-base-sentiment-latest.pth'))

use_cuda = True
if use_cuda:
    promptModel = promptModel.cuda()

promptModel.eval()
predict_result = []
with torch.no_grad():
    count = 0
    for batch in data_loader:
        if use_cuda:
            batch = batch.cuda()
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        predict_result.extend(preds.tolist())
        # print(classes[preds])
        
# 遍历 DataFrame 的每一行，并给 relative 列赋值为 1
for index, row in df.iterrows():
     df.at[index, 'relative'] = int(predict_result[index])
df[['id','full_text','relative']].to_csv('/home/linzhisheng/Digital_Research/data/filter_model_result_roberta.csv',index=False)