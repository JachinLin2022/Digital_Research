import pandas as pd
import re
from sklearn.metrics import f1_score,precision_score, recall_score, accuracy_score
import json
# 处理chatgpt标注后数据
def process_chat_label():
    df = pd.read_csv('data/tweet_candidate.csv')
    print(df)
    # df = df.dropna(subset=['chat_label'])
    df['label2'] = ''
    df['chat_label1'] = ''
    df['chat_label2'] = ''
    df['source'] = 'twitter'
    for idx,row in df.iterrows():
        chat_label = row['chat_label']
        try:
            parse = json.loads(chat_label)
            df.at[idx,'chat_label1'] = parse['label1'].lower()
            df.at[idx,'chat_label2'] = parse['label2'].lower()
            # print(parse)
        except Exception:
            print('error')

    # df = df.dropna(subset=['chat_label1'])
    df = df[df['chat_label1'].isin(['positive','negative','neutral'])]
    # print(df.columns)
    
    print(df)
    # df = df[df['full_text'].str.lower().str.contains('bitcoin') == 0]
    # print(df['full_text'])
    df.to_csv('data/tweet_candidate.csv',index=False)

    print(df['chat_label1'].value_counts())
    print(df['chat_label2'].value_counts())

def filter_reddit():
    reddit = pd.read_csv('reddit/bitcoin_posts_new.csv')
    print(reddit['id'].value_counts())


def concat_csv():
    import os
    folder_path = 'reddit'
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    print(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['id'])
    merged_df[['id','full_text']].to_csv('reddit.csv',index=False)
    print(merged_df)

def eval_step_chat():
    data = pd.read_csv('filter_model_manual_400.csv')
    target = pd.read_csv('steps_result2.csv')
    target['label'] = data['label']

    # target['analysis'] = target['analysis'].apply(lambda x: re.sub('[^01]', '', str(x)))
    # target['analysis'] = target['analysis'].apply(lambda x: x[0] if len(x)>0 else 1)
    # target['analysis'] = target['analysis'].astype(int)
    target['label'] = target['label'].astype(int)

    # count = sum(target['label'] == target['analysis'])

    # print(count/len(target))

    # print(target['analysis'].value_counts())


    # target.to_csv('step_result.csv',index=False)
    # process_chat_label()
    preds = []
    labels = []
    for idx,row in target.iterrows():
        true_label = row['label']
        label = 0
        history = json.loads(row['steps_result'])
        
        mention = history[3]['content'].lower()
        if mention.find('yes') >=0:
            stetement = history[5]['content'].lower()     
            if stetement.find('yes') >= 0:
                news = history[7]['content'].lower()   
                ad = history[9]['content'].lower()   
                objective = history[11]['content'].lower()   
                forecast = history[13]['content'].lower()   

                if news.find('yes')>=0:
                    label = 1
                if objective.find('yes')>=0:
                    label = 1
                if forecast.find('yes')>=0:
                    label = 1
                if ad.find('yes')>=0:
                    label = 0
        
            if stetement.find('no') >= 0:
                personal = history[7]['content'].lower()   
                news = history[9]['content'].lower()   
                trend = history[11]['content'].lower()   

                if news.find('yes') >= 0:
                    label = 1
                if trend.find('yes') >= 0:
                    label = 1
                if personal.find('yes') >= 0:
                    label = 0
                
        # print((label, true_label))
        preds.append(label)
        labels.append(true_label)
        target.at[idx,'cot_label'] = label
        # target.at[idx,'steps_result'] = json.dumps(json.loads(row['steps_result']),indent=4)
        # print(history)
        # break

    import numpy as np
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    f1 = f1_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np)
    recall = recall_score(labels_np, preds_np)
    accuracy = accuracy_score(labels_np, preds_np)
    print(f"F1-score: {f1:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"Accuracy: {accuracy:.5f}")

    print(preds.count(0))
    print(preds.count(1))

    target.to_csv('result.csv',index=False)
    

def is_english(text):
    import string
    """判断文本是否为英文"""
    # 忽略掉单词数等于0的文本
    if len(text) == 0:
        return False
    english_letters = set(string.ascii_letters)

    # 统计文本中英文字母的比例
    english_letters = [char for char in text if char.isalpha() and char in english_letters]
    english_ratio = len(english_letters) / len(text)
    
    return english_ratio > 0.5

def filter_by_num():

    
    
    
    
    tweets = pd.read_csv('tweets_sample_2020_2023.csv')
    tweets.drop_duplicates(subset=['id'], keep='first', inplace=True)
    COUNT_THRO = 100
    tweets = tweets[(tweets['user_follower_count'] > 1000) | (tweets['quote_count'] > COUNT_THRO) | (tweets['favorite_count'] > COUNT_THRO) | (tweets['reply_count'] > COUNT_THRO) | (tweets['retweet_count'] > COUNT_THRO)]
    tweets = tweets[tweets['full_text'].apply(lambda x: is_english(x))]

    
    
    # tweets.to_csv('tweet/tweet.csv',index=False)
    print(tweets)
    
def process_all_tweet():
    COUNT_THRO = 100
    import os
    files = os.listdir('tweet/day')

    for idx,file in enumerate(files):
        tweets = pd.read_csv('tweet/day/' + file)
        if len(tweets) == 0:
            continue
        # tweets.drop_duplicates(subset=['id'], keep='first', inplace=True)
        # tweets.drop_duplicates(subset=['full_text'], keep='first', inplace=True)
        # COUNT_THRO = 100
        # tweets = tweets[(tweets['user_follower_count'] > 10000) | (tweets['quote_count'] > COUNT_THRO) | (tweets['favorite_count'] > COUNT_THRO) | (tweets['reply_count'] > COUNT_THRO) | (tweets['retweet_count'] > COUNT_THRO)]
        # tweets = tweets[tweets['full_text'].apply(lambda x: is_english(x))]
        # tweets = tweets.sample(n=10)
        # tweets['period'] = file[7:-4]
        # tweets = tweets[tweets['full_text'].apply(lambda x: x.lower().find('bitcoin') >= 0)]
        # tweets.drop_duplicates(subset=['id'], keep='first', inplace=True)
        # tweets = tweets[(tweets['user_follower_count'] > 1000) | (tweets['quote_count'] > COUNT_THRO) | (tweets['favorite_count'] > COUNT_THRO) | (tweets['reply_count'] > COUNT_THRO) | (tweets['retweet_count'] > COUNT_THRO)]
        # tweets = tweets[tweets['full_text'].apply(lambda x: is_english(x))]
        # tweets['source'] = 'twitter'
        if idx == 0:
            res = tweets
        else:
            res = pd.concat([res,tweets])
    
    
    res['second'] = pd.to_datetime(res['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    
    res['month'] = res['second'].dt.year.astype(str) + '-' + res['second'].dt.month.astype(str)
    
    res['day'] = res['second'].dt.year.astype(str) + '-' + res['second'].dt.month.astype(str) + '-' + res['second'].dt.day.astype(str)
    
    res['second'] = res['second'].astype('int64') // 10**9
    res = res.sort_values('second', ascending=True)
    
    
    res.drop_duplicates(subset=['id'], keep='first', inplace=True)
    res.drop_duplicates(subset=['full_text'], keep='first', inplace=True)
    print(res)
    res.to_csv('tweets_raw_2020_2023.csv',index=False)
    


def filter_by_date():
    source = pd.read_csv('data/tweets_2020_2023.csv')
    
    filter = pd.read_csv('data/tweet_filter_2020_2023.csv')
    
    res = pd.merge(filter, source[['id','created_at']],on='id')
    res['created_at'] = res['created_at'].astype(str)
    res['period'] = res['period'].astype(str)
    print(res.columns)
    res = res[res.apply(lambda x: x['created_at'].find(x['period'].split('_')[0]) >=0, axis=1)]
    
    
    
    # print(res['created_at'])
    res.to_csv('data/tweet_candidate.csv',index=False)

def process():
    data = pd.read_csv('tweets_sample_2020_2023_filter.csv')
    # data['created_att'] = pd.to_datetime(data['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    # data['created_att'] = data['created_att'].astype('int64') // 10**9
    # data = data.sort_values('created_att', ascending=True)
    # data.to_csv('test.csv')
    data['full_text'] = data['full_text'].astype(str)
    print(data)
    data = data[data['full_text'].apply(lambda x: x.lower().find('bitcoin') >= 0)]
    print(data)
    data.to_csv('tweets_sample_2020_2023_filter.csv',index=False)

def get_tweets_by_year():
    input = pd.read_csv('tweets_raw_2020_2023.csv')
    input = input[input['month'].str.find('2021') >= 0]
    input.to_csv('tweets_raw_2021.csv',index=False)
    print(input)


# filter_by_num()
# process_chat_label()
# process_all_tweet()
# filter_by_date()
# process()
get_tweets_by_year()
# def get_label(x):
#     if x in ['approval', 'optimism','joy/happy']:
#         return 'positive'
#     elif x in ['fear','angry','sad']:
#         return 'nagetive'
#     else:
#         return 'neutral'

# pd.read_excel
# data = pd.read_excel('tweets_sample_2020_2023_filter.xlsx',sheet_name='Sheet1')
# data = data[['full_text','label2']]
# data = data.dropna(subset=['label2'])
# data['label2'] = data['label2'].apply(lambda x: x[:x.find(' ')])
# data['label2'] = data['label2'].apply(get_label)
# data.rename(columns={'full_text': 'text', 'label2': 'label'}, inplace=True)

# with open('train_data.jsonl', 'w') as f:
#     for index, row in data.iterrows():
#         json_line = row.to_json() + '\n'
#         f.write(json_line)
# print(data)