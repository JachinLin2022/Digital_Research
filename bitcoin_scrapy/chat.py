# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import pandas as pd
import os
import time
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import openai
key = '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'
openai.api_key = key

# batch_size = 10

tweets = pd.read_csv('hard_negative_test_with_label.csv')
# PROMPT = f'Help me label the above text, if it smay affect the price of bitcoin, label 1 else label 0. Please only output 0 or 1 for me.'
# PROMPT = f'Help me label the above text, if it related to the price trend of Bitcoin, label 1 else label 0. Please only output 0 or 1 for me.'
# PROMPT = 'Help me label the above text, if it related to the price trend of Bitcoin, label 1 else label 0. Please only output 0 or 1 for me. Here are some hints for you. The hinted sentence is a hard negative sample and its label is opposite to the sentence you need to annotate. Its content is:\n[\n{}\n]\n.'
PROMPT = 'Help me label the above text, if it related to the price trend of Bitcoin, label 1 else label 0. Please only output 0 or 1 for me. Here are some hints for you. The hinted sentences are hard negative samples and their labels are opposite to the sentence you need to annotate. Their contents are:\n[{}]\n.'
PROMPT = 'You are a text classifier and need to determine whether a text is related to the fluctuation of bitcoin prices. If related, label 1 else label 0. Your training set consists of:\n{}\nPlease help me label this text based on the training set:[{}]\nPlease only output 0 or 1 for me.'
types = ['random', 'max_same','min_same','max_diff','min_diff']
# types = ['random']
nums = [1,5,10,20,30]
# nums = [1,5,10,20,30]

for type in types:
    for num in nums:
        tweets[f'{type}_{num}_chat_label'] = ''

        for idx,row in tweets.iterrows():

            # if idx <= 172:
            #     continue
            # content = f"{row['full_text']}" + '\n' + PROMPT.format(row[f'{type}_sample_{num}'])
            content = PROMPT.format(row[f'{type}_sample_{num}'], row['full_text'])
            print(content)
            while(1):
                try:
                    rsp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                {"role": "user", "content": content},
                            ]
                    )
                    break
                except Exception as e:
                    print(e)

            result = rsp['choices'][0]['message']['content']
            tweets.at[idx, f'{type}_{num}_chat_label'] = str(result)
            if (idx+1) % 10 == 0:
                print('save')
                # tweets.to_csv('hard_negative_test_with_label.csv',index=False)
            print(type, num, idx, result)
            # time.sleep(1)
            
            # content = content + f"{idx}. {row['full_text']}" + '\n'
            # if (idx+1)%batch_size == 0:
            #     content = content + PROMPT
            #     print(content)
            #     # content = '#Bitcoin bouncing 🚀\n' + PROMPT
            #     rsp = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #             {"role": "user", "content": content},
            #         ]
            #     )
            #     result = rsp['choices'][0]['message']['content']
            #     print(result)
            #     # for res in result.split('\n'):
            #     #     idx = res.split(' - ')[0]
            #     #     label = res.split(' - ')[1]
            #     #     print(idx, label)

            #     content = ''








