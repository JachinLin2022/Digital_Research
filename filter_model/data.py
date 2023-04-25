import pandas as pd

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
