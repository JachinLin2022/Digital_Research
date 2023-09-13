import requests
import os
import json
import time
import pandas as pd
import threading
import calendar
import httpx
from urllib import parse
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
# output = './tweet/tweets_2020_0501_0530_live.csv'
# query = 'bitcoin until:2020-05-30 since:2020-05-01 -filter:replies lang:en'
# os.environ['http_proxy'] = 'http://127.0.0.1:9999'
# os.environ['https_proxy'] = 'http://127.0.0.1:9999'
# 创建线程事件
event = threading.Event()
DONE_LIST = []
def task_new(token, year, month, day, limit, sleep = 0):

    output = f'./tweet/new_day/tweets_{year}_{month+1}_{day+1}.csv'
    # output = f'./tweet/day/test.csv'
    if output in DONE_LIST:
        return
    exist = 0
    if os.path.exists(output):
        # try:
        df = pd.read_csv(output)
        # except:
        #     print(output)
        #     return 0
        exist = 1
        if len(df) > limit:
            return 1
    else:
        df = pd.DataFrame({}, columns=[
            'id', 'full_text',
            'created_at',
            'favorite_count',
            'quote_count',
            'reply_count',
            'retweet_count',
            'user_id',
            'user_name',
            'user_screen_name',
            'user_description',
            'user_friends_count',
            'user_follower_count',
            'user_favorite_count',
            'user_media_count',
            'cursor',
            'period'
        ])
        exist = 0




    if day == calendar.monthrange(year, month+1)[1] - 1:
        until = f'{year}-{month+2}-{1}'
        if month + 2 > 12:
            until = f'{year + 1}-{1}-{1}'
    else:
        until = f'{year}-{month+1}-{day+2}'
        # continue

    query = f'bitcoin until:{until} since:{year}-{month+1}-{day+1} -filter:replies lang:en'
    

    features = {"rweb_lists_timeline_redesign_enabled":'true',"responsive_web_graphql_exclude_directive_enabled":'true',"verified_phone_label_enabled":'false',"creator_subscriptions_tweet_preview_api_enabled":'true',"responsive_web_graphql_timeline_navigation_enabled":'true',"responsive_web_graphql_skip_user_profile_image_extensions_enabled":'false',"tweetypie_unmention_optimization_enabled":'true',"responsive_web_edit_tweet_api_enabled":'true',"graphql_is_translatable_rweb_tweet_is_translatable_enabled":'true',"view_counts_everywhere_api_enabled":'true',"longform_notetweets_consumption_enabled":'true',"responsive_web_twitter_article_tweet_consumption_enabled":'false',"tweet_awards_web_tipping_enabled":'false',"freedom_of_speech_not_reach_fetch_enabled":'true',"standardized_nudges_misinfo":'true',"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":'true',"longform_notetweets_rich_text_read_enabled":'true',"longform_notetweets_inline_media_enabled":'true',"responsive_web_media_download_video_enabled":'false',"responsive_web_enhance_cards_enabled":'false'}
    variables = {"rawQuery":query,"count":100,"querySource":"typed_query","product":"Latest",'cursor':''}
    if exist and len(df) > 0:
        last_row = df.tail(1)
        last_cursor_val = last_row['cursor'].values[0]
        variables['cursor'] = last_cursor_val
        # print(variables['cursor'])
        # print('读取上次爬取位置')
        if last_row['period'].values[0] == 'end' or last_row['period'].values[0] == 'no_entries':
            DONE_LIST.append(output)
            print('end')
            return
    print(query)
    # 账号token，可登录后在cookie中找到
    auth_token = token
    # 高级查询语句
    q = '(from:elonmusk)'
    url = f'https://twitter.com/i/api/2/search/adaptive.json?include_profile_interstitial_type=1&include_blocking=1&include_blocked_by=1&include_followed_by=1&include_want_retweets=1&include_mute_edge=1&include_can_dm=1&include_can_media_tag=1&include_ext_has_nft_avatar=1&include_ext_is_blue_verified=1&include_ext_verified_type=1&include_ext_profile_image_shape=1&skip_status=1&cards_platform=Web-12&include_cards=1&include_ext_alt_text=true&include_ext_limited_action_results=false&include_quote_count=true&include_reply_count=1&tweet_mode=extended&include_ext_views=true&include_entities=true&include_user_entities=true&include_ext_media_color=true&include_ext_media_availability=true&include_ext_sensitive_media_warning=true&include_ext_trusted_friends_metadata=true&send_error_codes=true&simple_quoted_tweet=true&q={parse.quote(q)}&query_source=typed_query&count=20&requestContext=launch&pc=1&spelling_corrections=1&include_ext_edit_control=true&ext=mediaStats%2ChighlightedLabel%2ChasNftAvatar%2CvoiceInfo%2CbirdwatchPivot%2Cenrichments%2CsuperFollowMetadata%2CunmentionInfo%2CeditControl%2Cvibe'
    headers = {
    'authority': 'twitter.com',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
    'cache-control': 'no-cache',
    'cookie': f'auth_token={auth_token};ct0=',
    'pragma': 'no-cache',
    'referer': 'https://twitter.com/',
    'sec-ch-ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'x-csrf-token': '', # ct0
    'x-twitter-active-user': 'yes',
    'x-twitter-auth-type': 'OAuth2Session',
    'x-twitter-client-language': 'zh-cn'
    }

    client = httpx.Client(headers=headers)
    # {"errors":[{"code":353,"message":"This request requires a matching csrf cookie and header."}]}
    for i in range(5):
        try:
            res1 = client.get(url=url.format())
            # 第一次访问用于获取response cookie中的ct0字段，并添加到x-csrf-token与cookie中
            ct0 = res1.cookies.get('ct0')
            # print(ct0)
            client.headers.update({
                'x-csrf-token': ct0,
                'cookie': f'auth_token={auth_token};ct0={ct0}'
            })
            break
        except Exception as e:
            if i == 4:
                print('获取token失败')
                return 0





    # url = 'https://twitter.com/i/api/2/search/adaptive.json'
    url = f'https://twitter.com/i/api/graphql/nK1dw4oV3k4w5TdtcAdSww/SearchTimeline'


        
    params = {
        'variables':json.dumps(variables),
        'features': json.dumps(features),
        'fieldToggles': json.dumps({"withArticleRichContentState":'false'})
    }
    # print(header)
    # urls = 'https://twitter.com/i/api/2/search/adaptive.json?include_profile_interstitial_type=1&include_blocking=1&include_blocked_by=1&include_followed_by=1&include_want_retweets=1&include_mute_edge=1&include_can_dm=1&include_can_media_tag=1&include_ext_has_nft_avatar=1&include_ext_is_blue_verified=1&include_ext_verified_type=1&include_ext_profile_image_shape=1&skip_status=1&cards_platform=Web-12&include_cards=1&include_ext_alt_text=true&include_ext_limited_action_results=false&include_quote_count=true&include_reply_count=1&tweet_mode=extended&include_ext_views=true&include_entities=true&include_user_entities=true&include_ext_media_color=true&include_ext_media_availability=true&include_ext_sensitive_media_warning=true&include_ext_trusted_friends_metadata=true&send_error_codes=true&simple_quoted_tweet=true&q=%23bitcoin&vertical=default&query_source=typd&count=20&requestContext=launch&pc=1&spelling_corrections=1&include_ext_edit_control=true&ext=mediaStats%2ChighlightedLabel%2ChasNftAvatar%2CvoiceInfo%2CbirdwatchPivot%2Cenrichments%2CsuperFollowMetadata%2CunmentionInfo%2CeditControl%2Cvibe'

    def fetch_url(url, params, client, num_retries=3):
        try:
            response = client.get(url=url,params=params)
        except:
            if num_retries > 0:
                time.sleep(1)
                print('retry')
                return fetch_url(url, params, client, num_retries-1)
            else:
                raise Exception('Failed to fetch url')
        if response.status_code != 200:
            print(response.content)
            if response.content.find('Rate') >= 0:
                raise Exception('rate limit')
            elif num_retries > 0:
                time.sleep(1)
                # print('retry2')
                return fetch_url(url, params, client, num_retries-1)
            else:
                raise Exception('Failed to fetch url')
        return response

    zero_count = 0
    res = []
    for i in range(9999):
        time.sleep(sleep)
        # if event.is_set():
        #     print('event_is_set')
        #     return 0
        if len(df) > limit:
            # df.to_csv(output, index=False)
            # print(df)
            break
        # r = requests.get(url=url,params=params,headers=header,verify=False)
        try:
            r = fetch_url(url, params, client)

        except Exception as e:
            # print(str(e))
            event.set()
            return 0
        print(f'{year}-{month+1}-{day+1}',r.status_code)
        msg = json.loads(r.content)
        # tweets = msg['globalObjects']['tweets']
        # users = msg['globalObjects']['users']


        
        last_cursor = variables['cursor']

        if 'search_by_raw_query' not in msg['data']:
            print('no search_by_raw_query')
            break
        for ins in msg['data']['search_by_raw_query']['search_timeline']['timeline']['instructions']:
            if ins['type'] == 'TimelineAddEntries':
                for entry in ins['entries']:
                    if entry['entryId'].find('cursor-bottom') >= 0:
                        cursor = entry['content']['value']
                        # print('first', cursor)
                        variables['cursor'] = cursor
                        params['variables'] = json.dumps(variables)
            elif ins['type'] == 'TimelineReplaceEntry':
                if ins['entry_id_to_replace'].find('cursor-bottom') >= 0:
                    cursor = ins['entry']['content']['value']
                    # print('update', cursor)
                    variables['cursor'] = cursor
                    params['variables'] = json.dumps(variables)

        # print(msg['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0].keys())
        if 'entries' not in msg['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]:
            zero_count = zero_count + 1
            if zero_count == 3:
                print('no entries')
                df.at[df.index[-1], 'period'] = 'no_entries'
                df.to_csv(output, index=False)
                return 1
            continue
        else:
            zero_count = 0


        
        entries = msg['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]['entries']



        if last_cursor == variables['cursor']:
            print('error, cursor not change')

        
        if len(entries) == 0:
            zero_count = zero_count + 1
            if zero_count == 3:
                print('no more tweets')
                df.at[df.index[-1], 'period'] = 'end'
                df.to_csv(output, index=False)
                return 1
            continue
        else:
            zero_count = 0
        

        for entry in entries:
            if entry['entryId'].find('tweet') >= 0:
                # print()
                try:
                    result = entry['content']['itemContent']['tweet_results']['result']
                    if 'tweet' in result:
                        tweet = result['tweet']['legacy']
                        user = result['tweet']['core']['user_results']['result']['legacy']
                    else:
                        tweet = result['legacy']
                        user = result['core']['user_results']['result']['legacy']
                    
                    scrapy_tweet = {
                            'id': tweet['conversation_id_str'],
                            'full_text': tweet['full_text'],
                            'created_at': tweet['created_at'],
                            'favorite_count': tweet['favorite_count'],
                            'quote_count': tweet['quote_count'],
                            'reply_count': tweet['reply_count'],
                            'retweet_count': tweet['retweet_count'],
                            'user_id': tweet['user_id_str'],
                            'user_name': user['name'],
                            'user_screen_name': user['screen_name'],
                            'user_description': user['description'],
                            'user_friends_count': user['friends_count'],
                            'user_follower_count': user['followers_count'],
                            'user_favorite_count': user['favourites_count'],
                            'user_media_count': user['media_count'],
                            'cursor': variables['cursor'],
                            'period': f'{year}-{month+1}'
                        }
                    # print(scrapy_tweet)
                    res.append(scrapy_tweet)
                    scrapy_tweet = pd.DataFrame([scrapy_tweet])
                    df = pd.concat([df, scrapy_tweet], ignore_index=True)
                except Exception as e:
                    f = open('error.txt','w')
                    print('error')
                    f.write(str(e))
                    f.write(json.dumps(entry,indent=4))
                    return 0

        if i > 0 or i % 2 == 0:
            df.to_csv(output, index=False)
            print(df)
        if len(df) > limit:
            df.to_csv(output, index=False)
            print(df)
            break

import random
class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            return "Queue is empty"

    def size(self):
        return len(self.items)
    
    def shuffle(self):
        random.shuffle(self.items)


def main():
    

    # token = queue.dequeue()
    # queue.enqueue(token)
    # task_new(token, 2020, 0, 0, 10000, 0)
    # if event.is_set():
    #     event.clear()
    #     return 0
    # threads = []
    for year in [2023]:
        # threads = []
        # thread_num = 31
        for month in range(12):
            if year == 2023 and month == 5:
                break
            # if month != 0:
            #     continue
            print(year,month)
            # for day in range(calendar.monthrange(year, month+1)[1]):
            #     token = queue.dequeue()
            #     queue.enqueue(token)
            #     t = threading.Thread(target=task_new, args=(token, year, month, day, 999999, 0))
            #     threads.append(t)
            #     t.start()
            
            # if month != 10:
            #     continue
            
            thread_num = 31
            for day in range(0, calendar.monthrange(year, month+1)[1], thread_num):
                numbers = list(range(day, day + thread_num))
                threads = []
                for number in numbers:
                    if number < calendar.monthrange(year, month+1)[1]:
                        # print(number)
                        token = queue.dequeue()
                        queue.enqueue(token)
                        t = threading.Thread(target=task_new, args=(token, year, month, number, 9999999, 0))
                        threads.append(t)
                        t.start()
                for t in threads:
                    t.join()

            #     print('next month')
        # time.sleep(60*5)
        # for t in threads:
        #     t.join()


    return 0

token_list = []
queue = Queue()
f = open('account.txt','r')
for row in f.readlines():
    # print(row.split('----')[-1])
    token_list.append(row.split('----')[-1][:-1])
    queue.enqueue(row.split('----')[-1][:-1])

print("Original Queue:", len(queue.items))

queue.shuffle()

print("Shuffled Queue:", len(queue.items))

# print(token_list)


while (1):
    # print('切换token', token_list[index])
    # time.sleep(60*15)


    r = main()
    print('sleep 30min')
    # time.sleep(60*5)


    # index = index + 1
    # if index == len(token_list):
        # time.sleep(60*3)
        # index = 0

    # if (r):
    #     break


