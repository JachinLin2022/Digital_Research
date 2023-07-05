import praw
import os
import pandas as pd
import datetime
import time
from pmaw import PushshiftAPI
# from praw.models.util import token_manager



os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

reddit = praw.Reddit(
    client_id="KVQGARREpAmG1edIruOYfw",
    client_secret="7YjaJZ2lny3nDZ8x3MkTWjF2UG0NYw",
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.39"
)

from pmaw import PushshiftAPI

api = PushshiftAPI()
comments = api.search_comments(subreddit="science", limit=1000)
comment_list = [comment for comment in comments]