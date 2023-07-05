import praw
import os
import pandas as pd
import datetime
import time
# from praw.models.util import token_manager



os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

reddit = praw.Reddit(
    client_id="KVQGARREpAmG1edIruOYfw",
    client_secret="7YjaJZ2lny3nDZ8x3MkTWjF2UG0NYw",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.39"
)


# # 创建 UrlTokenManager 对象
# token_manager = token_manager.UrlTokenManager(reddit)

# # 设置搜索范围（UTC 时间）
# start_time = datetime.datetime(2023, 5, 1, 0, 0)
# end_time = datetime.datetime(2023, 5, 10, 0, 0)


subreddit = reddit.subreddit("bitcoin")
 
# Define the number of posts to retrieve per request
post_limit = 1000

# Define the number of requests to make to retrieve the desired number of posts
num_requests = 1

# Define an empty list to store the post data
post_list = []
after = ''

# Loop through the requested number of times to retrieve posts
for i in range(num_requests):
    # Get the next batch of posts
    print(after)
    # posts = subreddit.search(query='bitcoin',limit=post_limit)
    posts = subreddit.new(limit=post_limit)
    # Keep track of the ID of the last post in the current batch
    # print(len(list(posts)))
    
    # Loop through each post and extract the relevant information
    for post in posts:
        created_at = datetime.datetime.fromtimestamp(post.created_utc)
        post_data = {
            'id': post.fullname,
            'title': post.title.replace("’","'"),
            'body': post.selftext.replace("’","'"),
            'full_text': post.title.replace("’","'") + '\n' + post.selftext.replace("’","'"),
            'author': post.author,
            'url': post.url,
            'permalink': 'https://www.reddit.com' + post.permalink,
            'score': post.score,
            # 'created_utc': post.created_utc,
            'created_utc': created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'num_comments': post.num_comments
        }
        post_list.append(post_data)
        after = post.fullname
    
    # Print the current number of posts retrieved
    print(f"Retrieved {len(post_list)} posts...")
    
    # If we have retrieved the desired number of posts, break out of the loop
    if len(post_list) >= post_limit*num_requests:
        break
df = pd.DataFrame(post_list)
df.to_csv('reddit/bitcoin_posts_new.csv', index=False)