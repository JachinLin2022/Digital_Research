import pandas as pd


def fitler_some_data():
    COUNT_THRO = 100
    tweets = pd.read_csv('/home/linzhisheng/Digital_Research/data/tweets_0401_0423_live.csv')
    tweets.drop_duplicates(subset=['id'], keep='first', inplace=True)
    tweets = tweets[(tweets['user_follower_count'] > 10000) | (tweets['quote_count'] > COUNT_THRO) | (tweets['favorite_count'] > COUNT_THRO) | (tweets['reply_count'] > COUNT_THRO) | (tweets['retweet_count'] > COUNT_THRO)]
    tweets = tweets.sort_values('user_follower_count', ascending=False)
    tweets = tweets.assign(relative = '')

    tweets.to_csv('filter_post.csv',index=False)
    # lable_tweets = tweets[1000:1100]
    # lable_tweets[["id","full_text","relative"]].to_csv('/home/linzhisheng/Digital_Research/data/filter_tweets_sample.csv',index=False)
    print(tweets)
    # print(tweets[['user_follower_count','user_screen_name']])




tweets = pd.read_excel('/home/linzhisheng/Digital_Research/data/bitcoin_posts_label.xlsx')
# # print(tweets)
for i in range(len(tweets)): 
    print(f"{i+1} {tweets.iloc[i]['title']} {tweets.iloc[i]['body']}")
# tweets = tweets.assign(chatgpt_label = "")
# tweets = tweets.assign(chatgpt_reason = "")
# import re

# # 打开文件并读取内容
# with open('1.sty', 'r') as f:
#     data = f.read()

# # 使用正则表达式按序号分割字符串，并存储到列表中
# data_list = re.findall(r'\d+\.\s+(.*)', data)


# for i,row in enumerate(data_list): 
#     label = row[7:8]
#     reason = row[9:]
#     tweets.at[i, 'chatgpt_label'] = label
#     tweets.at[i, 'chatgpt_reason'] = reason
#     # print(i)  # 输出列表

# tweets.to_csv('label_tweets.csv',index=False)