# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import openai
import threading
key = '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'
openai.api_key = key

# batch_size = 10

tweets = pd.read_csv('tweets_sample_2020_2023_filter.csv')
# PROMPT = f'Help me label the above text, if it smay affect the price of bitcoin, label 1 else label 0. Please only output 0 or 1 for me.'
# PROMPT = f'Help me label the above text, if it related to the price trend of Bitcoin, label 1 else label 0. Please only output 0 or 1 for me.'
# PROMPT = 'Help me label the above text, if it related to the price trend of Bitcoin, label 1 else label 0. Please only output 0 or 1 for me. Here are some hints for you. The hinted sentence is a hard negative sample and its label is opposite to the sentence you need to annotate. Its content is:\n[\n{}\n]\n.'
# PROMPT = 'Help me label the above text, if it related to the price trend of Bitcoin, label 1 else label 0. Please only output 0 or 1 for me. Here are some hints for you. The hinted sentences are hard negative samples and their labels are opposite to the sentence you need to annotate. Their contents are:\n[{}]\n.'
# PROMPT = 'You are a text classifier and need to determine whether a text is related to the fluctuation of bitcoin prices. If related, label 1 else label 0. Your training set consists of:\n{}\nPlease help me label this text based on the training set:[{}]\nPlease only output 0 or 1 for me.'
# PROMPT = '''
# "{}"
# Follow the steps below to conduct sentiment analysis on the Bitcoin-related text in double quotes about the Bitcoin price.
# First, choose one of these three words: [Positive, Negative, Neutral]
# Second, if it is positive, then select another word from these words: [Approval, Optimism, Other]. If it is negative, then select another word from these words: [Fear, Angry, Other]. If it is neutral, select [Other].
# Please output it in json format. Keys are [label1, label2].
# '''
PROMPT = '''
将以下内容翻译成中文(简体)语言：
{}
'''
target = 'chinese'
# tweets[target] = ''
tweets[target] = tweets[target].astype(str)

def chat(content):
    while(1):
        try:
            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "user", "content": content},
                    ],
                timeout = 1
            )
            break
        except Exception as e:
            print(e)
    return rsp



def process_elements(start, end):
    for i in range(start, end):
        if len(tweets.loc[i,target]) >=5:
            continue
        content = PROMPT.format(tweets.loc[i,'full_text'])
        rsp = chat(content)
        result = rsp['choices'][0]['message']['content']
        tweets.at[i,target] = str(result)
        print(i, result)
        if (i+1) % 5 ==0:
            print('save')
            tweets.to_csv('tweets_sample_2020_2023_filter.csv',index=False)
    print('done===============================')


print(tweets)

threads = []
num_threads = 10
batch_size = len(tweets)/num_threads
for i in range(num_threads):
    # 计算当前线程要处理的元素范围
    start_index = int(i * batch_size)
    end_index = int(min(start_index + batch_size, len(tweets)))
    # print(start_index, end_index)
    # 创建线程并启动
    t = threading.Thread(target=process_elements, args=(start_index,end_index,))
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()



print('done')
tweets.to_csv('tweets_sample_2020_2023_filter.csv',index=False)



# for idx,row in tweets.iterrows():

#     content = PROMPT.format(row['full_text'])
#     t = threading.Thread(target=chat,args=(content,))

#     with ThreadPoolExecutor(max_workers=5) as executor:
#         results = []
#         for i in range(5):
#             result = executor.submit(chat, content)
#             results.append(result)

#         for future in results:
#             result = future.result()['choices'][0]['message']['content']
#             print(result)
    
#     break
    # if idx > 321:
    #     continue
    # if len(row[target]) >=5:
    #     # print(row['chat_label'])
    #     continue
    # # content = f"{row['full_text']}" + '\n' + PROMPT.format(row[f'{type}_sample_{num}'])
    # content = PROMPT.format(row['full_text'])
    # # print(content)
    # while(1):
    #     try:
    #         rsp = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                     {"role": "user", "content": content},
    #                 ],
    #             timeout = 1
    #         )
    #         break
    #     except Exception as e:
    #         print(e)
    # result = rsp['choices'][0]['message']['content']
    # tweets.at[idx,target] = str(result)
    # print(idx, result)
    
    # content = PROMPT2.format(row['full_text'])
    # while(1):
    #     try:
    #         rsp = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                     {"role": "user", "content": content},
    #                 ]
    #         )
    #         break
    #     except Exception as e:
    #         print(e)
    # result = rsp['choices'][0]['message']['content']
    # tweets.at[idx, f'chinese'] = str(result)
    # print(idx, result)
    
    
    # if (idx+1) % 2 == 0:
    #     print('save')
    #     tweets.to_csv('data/tweet-2020-2023.csv',index=False)