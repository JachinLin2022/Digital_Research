import requests
import os
import json
import time
import pandas as pd
# from selenium import webdriver
from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import threading
import calendar

os.environ['http_proxy'] = 'http://127.0.0.1:9999'
os.environ['https_proxy'] = 'http://127.0.0.1:9999'

import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
# output = './tweet/tweets_2020_0501_0530_live.csv'
# query = 'bitcoin until:2020-05-30 since:2020-05-01 -filter:replies lang:en'

# 创建线程事件
event = threading.Event()


def task(HEADER, year, month, day, until, limit):
    output = f'./tweet/day/tweets_{year}_{month+1}_{day+1}.csv'
    query = f'bitcoin until:{until} since:{year}-{month+1}-{day+1} -filter:replies lang:en'
    print(query)
    exist = 0
    if os.path.exists(output):
        df = pd.read_csv(output)
        exist = 1
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

    url = 'https://twitter.com/i/api/2/search/adaptive.json'

    params = {
        'include_profile_interstitial_type': '1',
        'include_blocking': '1',
        'include_blocked_by': '1',
        'include_followed_by': '1',
        'include_want_retweets': '1',
        'include_mute_edge': '1',
        'include_can_dm': '1',
        'include_can_media_tag': '1',
        'include_ext_has_nft_avatar': '1',
        'include_ext_is_blue_verified': '1',
        'include_ext_verified_type': '1',
        'include_ext_profile_image_shape': '1',
        'skip_status': '1',
        'cards_platform': 'Web-12',
        'include_cards': '1',
        'include_ext_alt_text': 'true',
        'include_ext_limited_action_results': 'false',
        'include_quote_count': 'true',
        'include_reply_count': '1',
        'tweet_mode': 'extended',
        'include_ext_views': 'true',
        'include_entities': 'true',
        'include_user_entities': 'true',
        'include_ext_media_color': 'true',
        'include_ext_media_availability': 'true',
        'include_ext_sensitive_media_warning': 'true',
        'include_ext_trusted_friends_metadata': 'true',
        'send_error_codes': 'true',
        'simple_quoted_tweet': 'true',
        'tweet_search_mode': 'live',
        # 'q': '#bitcoin until:2022-04-02 since:2022-04-01 lang:en',
        'q': query,
        # 'cursor': 'DAACCgACFuZuD_-AJxAKAAMW5m4P_3_Y8AgABAAAAAILAAUAAADoRW1QQzZ3QUFBZlEvZ0dKTjB2R3AvQUFBQUJNVTlMVkp3RmNRQWhUMHZleUNWeEFIRlBTMUV2N1hzQUlVOUxXM3psWlFBeFQwdHlEcUZYQUVGUFN0VUd5VlVBQVU5TGx2VzVmUUNSVDB0VldUbWpBRkZQUzRBV0VYc0FrVTlNQkhHVlJ3QVJUMHVRNUVGWUFDRlBTdHZDblhvQUVVOUx0ZjZCV0FBQlQwclBKbUY4QUdGUFMvakFZWDBBTVU5TFRMUXRSZ0F4VDB2TEZRMUdBR0ZQU3d4QmxYMEFVVTlMOFdsQldRQkE9PQgABgAAAAAIAAcAAAAADAAICgABFPSs8mYXwAYAAAA',
        'vertical': 'default',
        'query_source': 'typed_query',
        'count': '100',
        'requestContext': 'launch',
        'pc': '1',
        'spelling_corrections': '1',
        'include_ext_edit_control': 'true',
        'ext': 'mediaStats,highlightedLabel,hasNftAvatar,voiceInfo,birdwatchPivot,enrichments,superFollowMetadata,unmentionInfo,editControl,vibe'
    }

    header = {
        'Host': 'twitter.com',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.48',
        'Accept-Encoding': 'gzip, deflate, br'
    }

    with open('header.txt', 'r') as file:
        for line in file:
            if line.startswith('cookie') or line.startswith('authorization') or line.startswith('x-csrf-token'):
                key, value = line.strip().split(": ")
                header[key] = value

    header = HEADER

    if exist and len(df) > 0:
        last_row = df.tail(1)
        last_cursor_val = last_row['cursor'].values[0]
        params['cursor'] = last_cursor_val
        print('读取上次爬取位置')
        # if last_cursor_val == 'end':
        #     print('end')
        #     return

    # print(header)
    # urls = 'https://twitter.com/i/api/2/search/adaptive.json?include_profile_interstitial_type=1&include_blocking=1&include_blocked_by=1&include_followed_by=1&include_want_retweets=1&include_mute_edge=1&include_can_dm=1&include_can_media_tag=1&include_ext_has_nft_avatar=1&include_ext_is_blue_verified=1&include_ext_verified_type=1&include_ext_profile_image_shape=1&skip_status=1&cards_platform=Web-12&include_cards=1&include_ext_alt_text=true&include_ext_limited_action_results=false&include_quote_count=true&include_reply_count=1&tweet_mode=extended&include_ext_views=true&include_entities=true&include_user_entities=true&include_ext_media_color=true&include_ext_media_availability=true&include_ext_sensitive_media_warning=true&include_ext_trusted_friends_metadata=true&send_error_codes=true&simple_quoted_tweet=true&q=%23bitcoin&vertical=default&query_source=typd&count=20&requestContext=launch&pc=1&spelling_corrections=1&include_ext_edit_control=true&ext=mediaStats%2ChighlightedLabel%2ChasNftAvatar%2CvoiceInfo%2CbirdwatchPivot%2Cenrichments%2CsuperFollowMetadata%2CunmentionInfo%2CeditControl%2Cvibe'

    def fetch_url(url, params, header, num_retries=10):
        try:
            response = requests.get(
                url=url, params=params, headers=header, verify=False, timeout=(5, 5))
        except:
            if num_retries > 0:
                time.sleep(1)
                print('retry')
                return fetch_url(url, params, header, num_retries-1)
            else:
                raise Exception('Failed to fetch url')
        if response.status_code != 200:
            print(response.content)
            if response.content.find('Rate') >= 0:
                raise Exception('rate limit')
            elif num_retries > 0:
                time.sleep(1)
                # print('retry2')
                return fetch_url(url, params, header, num_retries-1)
            else:
                raise Exception('Failed to fetch url')
        return response

    zero_count = 0
    res = []
    for i in range(9999):
        if event.is_set():
            print('event_is_set')
            return 0
        if len(df) > limit:
            # df.to_csv(output, index=False)
            # print(df)
            break
        # r = requests.get(url=url,params=params,headers=header,verify=False)
        try:
            r = fetch_url(url, params, header)
        except:
            event.set()
            return 0
        print(f'{year}-{month+1}-{day+1}',r.status_code)
        msg = json.loads(r.content)
        tweets = msg['globalObjects']['tweets']
        users = msg['globalObjects']['users']

        if len(tweets) == 0:
            zero_count = zero_count + 1
        else:
            zero_count = 0
        if zero_count == 10:
            print('no more tweets')
            # df.at[df.index[-1], 'cursor'] = 'end'
            # df.to_csv(output, index=False)
            break

        # print(users.keys())

        # 获取下一个分页的cursor
        for ins in msg['timeline']['instructions']:
            if 'addEntries' in ins:
                cursors = ins['addEntries']['entries']
                for cursor in cursors:
                    if cursor['entryId'].find('cursor-bottom') >= 0:
                        # print(cursor['content']['operation']['cursor']['value'])
                        params['cursor'] = cursor['content']['operation']['cursor']['value']
            else:
                cursors = ins['replaceEntry']['entry']
                if cursors['entryId'].find('cursor-bottom') >= 0:
                    params['cursor'] = cursors['content']['operation']['cursor']['value']

        # 获取每条帖子的信息
        for tweet_time_line in tweets.keys():
            tweet = tweets[tweet_time_line]

            # 获取发帖用户的信息
            user_id_str = tweet['user_id_str']
            user = users[user_id_str]
            # LIMIT = 100
            # if int(tweet['favorite_count']) < LIMIT and int(tweet['quote_count']) < LIMIT and int(tweet['retweet_count']) < LIMIT and int(tweet['reply_count']) < LIMIT:
            #     print(tweet_time_line)
            #     continue
            scrapy_tweet = {
                'id': tweet_time_line,
                'full_text': tweet['full_text'],
                'created_at': tweet['created_at'],
                'favorite_count': tweet['favorite_count'],
                'quote_count': tweet['quote_count'],
                'reply_count': tweet['reply_count'],
                'retweet_count': tweet['retweet_count'],
                'user_id': user_id_str,
                'user_name': user['name'],
                'user_screen_name': user['screen_name'],
                'user_description': user['description'],
                'user_friends_count': user['friends_count'],
                'user_follower_count': user['followers_count'],
                'user_favorite_count': user['favourites_count'],
                'user_media_count': user['media_count'],
                'cursor': params['cursor'],
                'period': f'{year}-{month+1}'
            }
            # print(scrapy_tweet)
            res.append(scrapy_tweet)
            scrapy_tweet = pd.DataFrame([scrapy_tweet])
            df = pd.concat([df, scrapy_tweet], ignore_index=True)

        if i > 0 and i % 2 == 0:
            df.to_csv(output, index=False)
            print(df)
        if len(df) > limit:
            df.to_csv(output, index=False)
            print(df)
            break

order = {
    'Switch to @jachinlin2': 'Switch to @LiliannaPe8191',
    'Switch to @LiliannaPe8191': 'Switch to @JaggerDura48575',
    'Switch to @JaggerDura48575': 'Switch to @jachinlin222559',
    'Switch to @jachinlin222559': 'Switch to @CeciliaRid3549',
    'Switch to @CeciliaRid3549': 'Switch to @jachinlin2'
}
def main():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    # options.add_argument('--disable-gpu')

    options.add_argument('--ignore-certificate-error')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('log-level=2')
    options.add_argument(
        '--user-data-dir=C:\\Users\\67093\\AppData\\Local\\Google\\Chrome\\User Data')
    browser = webdriver.Chrome(options=options)
    # browser.minimize_window()  # 将窗口最小化

    browser.set_window_rect(0, 0, 800, 1000)
    browser.get('https://twitter.com/search?q=bitcoin&src=typed_query&f=top')
    # time.sleep(1.5)
    # browser.refresh()
    # print(len(browser.requests))
    while 1:
        try:
            button = WebDriverWait(browser, 20).until(expected_conditions.visibility_of_element_located(
                    (By.XPATH, '//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[2]/div/div')))
            button.click()
            time.sleep(3)
            roles = WebDriverWait(browser, 20).until(expected_conditions.presence_of_all_elements_located(
            (By.XPATH, '//*[@id="layers"]/div[2]/div/div/div[2]/div/div[2]/div/div/div/div/div/div[1]')))
            break
        except:
            print('refresh')
            browser.refresh()


    
    
    try:
        
        # button = WebDriverWait(browser, 10).until(expected_conditions.visibility_of_element_located(
        #     (By.XPATH, '//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[2]/div/div')))
        # button.click()
        # time.sleep(3)
        
        parent_element = browser.find_element(By.XPATH, '//*[@id="layers"]/div[2]/div/div/div[2]/div/div[2]/div/div/div/div/div/div[1]')
        child_elements = parent_element.find_elements(By.XPATH,'./div')
        current = browser.find_element(By.XPATH, '//*[@id="layers"]/div[2]/div/div/div[2]/div/div[2]/div/div/div/div/div/li/div/div[2]/div/div/div/div[2]/div/div/div/span').text
        print(current,len(child_elements))

        for element in child_elements:
            name = element.get_attribute('aria-label')
            # print(name)
            if name == order['Switch to ' + current]:
                print(name)
                if name == 'Switch to @jachinlin2':
                    time.sleep(60*5)
                print(len(browser.requests))
                element.click()
                break
        
        # next_role = WebDriverWait(browser, 30).until(expected_conditions.visibility_of_element_located(
        #     (By.XPATH, '//*[@id="layers"]/div[2]/div/div/div[2]/div/div[2]/div/div/div/div/div/div[1]')))
        # next_role.click()
    except Exception as e:
        print(str(e))
        return 0
    HEADER = ''
    time.sleep(5)
    print(len(browser.requests))
    for request in browser.requests:
        if request.response and "x-csrf-token" in request.headers and 'twitter.com/i/api/2/' in request.url:
            # print(request.url)
            print(request)
            HEADER = request.headers
            # print(request.headers)
            break
    
    browser.quit()
    # exit(0)
    if HEADER == '':
        return 0
    
    # task(HEADER, 2020, 9, 0, '2020-10-2', 1000)
    for year in [2020, 2021, 2022, 2023]:
        for month in range(12):
            if year == 2023 and month == 5:
                exit(0)
            threads = []
            # print(year,month)
            for day in range(calendar.monthrange(year, month+1)[1]):
                if day == calendar.monthrange(year, month+1)[1] - 1:
                    until = f'{year}-{month+2}-{1}'
                    if month + 2 > 12:
                        until = f'{year + 1}-{1}-{1}'
                else:
                    until = f'{year}-{month+1}-{day+2}'
                    # continue
                t = threading.Thread(target=task, args=(HEADER, year, month, day, until,1000))
                threads.append(t)
                t.start()
                # task(HEADER, year, month, day, until, 1000)
                # if event.is_set():
                #     event.clear()
                #     return 0
            for t in threads:
                t.join()
            if event.is_set():
                event.clear()
                return 0
    # df.to_csv(output, index=False)
    # print(df)
    # return 0
    return 1


while (1):
    r = main()
    # time.sleep(60)
    if (r):
        break
    print('切换账号')
