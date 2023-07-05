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

os.environ['http_proxy'] = 'http://127.0.0.1:9999'
os.environ['https_proxy'] = 'http://127.0.0.1:9999'



def main():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-error')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('log-level=2')
    options.add_argument('--user-data-dir=C:\\Users\\Jachin\\AppData\\Local\\Google\\Chrome\\User Data')
    browser = webdriver.Chrome(options=options)
    browser.set_window_rect(0,0,800,1000)
    browser.get('https://twitter.com/search?q=bitcoin&src=typed_query&f=top')

    try:
        button = WebDriverWait(browser, 30).until(expected_conditions.visibility_of_element_located((By.XPATH, '//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[2]/div/div')))
        button.click()
        time.sleep(3)
        next_role = WebDriverWait(browser, 30).until(expected_conditions.visibility_of_element_located((By.XPATH, '//*[@id="layers"]/div[2]/div/div/div[2]/div/div[2]/div/div/div/div/div/div[1]')))
        next_role.click()
        time.sleep(5)
    except:
        return 0
    HEADER = ''
    for request in browser.requests:
        if request.response and "x-csrf-token" in request.headers and 'twitter.com/i/api/2/' in request.url:
            print(request.url)
            HEADER = request.headers
            # print(request.headers)
            break
    browser.quit()
    if HEADER == '':
        return 0
    
    output = './tweet/tweets_user.csv'
    exist = 0
    if os.path.exists(output):
        df = pd.read_csv(output)
        exist = 1
    else:
        df = pd.DataFrame({}, columns=[
            'user_id',
            'user_name',
            'user_screen_name',
            'user_description',
            'user_friends_count',
            'user_follower_count',
            'user_favorite_count',
            'user_media_count',
            'cursor'
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
        'include_ext_media_color':'true',
        'include_ext_media_availability':'true',
        'include_ext_sensitive_media_warning':'true',
        'include_ext_trusted_friends_metadata':'true',
        'send_error_codes':'true',
        'simple_quoted_tweet':'true',
        'tweet_search_mode':'live',
        'q': 'bitcoin',
        'result_filter': 'user',
        # 'cursor': 'DAACCgACFuZuD_-AJxAKAAMW5m4P_3_Y8AgABAAAAAILAAUAAADoRW1QQzZ3QUFBZlEvZ0dKTjB2R3AvQUFBQUJNVTlMVkp3RmNRQWhUMHZleUNWeEFIRlBTMUV2N1hzQUlVOUxXM3psWlFBeFQwdHlEcUZYQUVGUFN0VUd5VlVBQVU5TGx2VzVmUUNSVDB0VldUbWpBRkZQUzRBV0VYc0FrVTlNQkhHVlJ3QVJUMHVRNUVGWUFDRlBTdHZDblhvQUVVOUx0ZjZCV0FBQlQwclBKbUY4QUdGUFMvakFZWDBBTVU5TFRMUXRSZ0F4VDB2TEZRMUdBR0ZQU3d4QmxYMEFVVTlMOFdsQldRQkE9PQgABgAAAAAIAAcAAAAADAAICgABFPSs8mYXwAYAAAA',
        'vertical': 'default',
        'query_source': 'typed_query',
        'count': '400',
        'requestContext': 'launch',
        'pc': '1',
        'spelling_corrections': '1',
        'include_ext_edit_control':'true',
        'ext': 'mediaStats,highlightedLabel,hasNftAvatar,voiceInfo,birdwatchPivot,enrichments,superFollowMetadata,unmentionInfo,editControl,vibe'
    }


    header = {
        'Host': 'twitter.com',
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.48',
        'Accept-Encoding': 'gzip, deflate, br'
    }

    with open('header.txt', 'r') as file:
        for line in file:
            if line.startswith('cookie') or line.startswith('authorization') or line.startswith('x-csrf-token'):
                key, value = line.strip().split(": ")
                header[key] = value

    header = HEADER
    
    if exist:
        last_row = df.tail(1)
        last_cursor_val = last_row['cursor'].values[0]
        params['cursor'] = last_cursor_val
        print('读取上次爬取位置')


    # print(header)
    # urls = 'https://twitter.com/i/api/2/search/adaptive.json?include_profile_interstitial_type=1&include_blocking=1&include_blocked_by=1&include_followed_by=1&include_want_retweets=1&include_mute_edge=1&include_can_dm=1&include_can_media_tag=1&include_ext_has_nft_avatar=1&include_ext_is_blue_verified=1&include_ext_verified_type=1&include_ext_profile_image_shape=1&skip_status=1&cards_platform=Web-12&include_cards=1&include_ext_alt_text=true&include_ext_limited_action_results=false&include_quote_count=true&include_reply_count=1&tweet_mode=extended&include_ext_views=true&include_entities=true&include_user_entities=true&include_ext_media_color=true&include_ext_media_availability=true&include_ext_sensitive_media_warning=true&include_ext_trusted_friends_metadata=true&send_error_codes=true&simple_quoted_tweet=true&q=%23bitcoin&vertical=default&query_source=typd&count=20&requestContext=launch&pc=1&spelling_corrections=1&include_ext_edit_control=true&ext=mediaStats%2ChighlightedLabel%2ChasNftAvatar%2CvoiceInfo%2CbirdwatchPivot%2Cenrichments%2CsuperFollowMetadata%2CunmentionInfo%2CeditControl%2Cvibe'


    def fetch_url(url, params, header, num_retries=5):
        try:
            response = requests.get(url=url,params=params,headers=header,verify=False, timeout=(5, 5))
        except:
            if num_retries > 0:
                time.sleep(1)
                print('retry')
                return fetch_url(url, params, header, num_retries-1)
            else:
                raise Exception('Failed to fetch url')
        if response.status_code != 200:
            print(response.content)
            if num_retries > 0:
                time.sleep(1)
                print('retry2')
                return fetch_url(url, params, header, num_retries-1)
            else:
                raise Exception('Failed to fetch url')
        return response


    zero_count = 0
    for i in range(1000):
        # r = requests.get(url=url,params=params,headers=header,verify=False)
        try:
            r = fetch_url(url, params, header)
        except:
            return 0
        print(r.status_code)
        msg = json.loads(r.content)
        users = msg['globalObjects']['users']
        
        
        if len(users) == 0:
            zero_count = zero_count + 1
        else:
            zero_count = 0
        if zero_count == 10:
            print('no more users')
            return 1
        
        # print(users.keys())

        # 获取下一个分页的cursor
        for ins in msg['timeline']['instructions']:

            if 'addEntries' in ins:
                cursors = ins['addEntries']['entries']
                for cursor in cursors:
                    if cursor['entryId'].find('cursor-bottom') >= 0:
                        # print(cursor['content']['operation']['cursor']['value'])
                        params['cursor'] = cursor['content']['operation']['cursor']['value']
            elif 'replaceEntry' in ins:
                cursors = ins['replaceEntry']['entry']
                if cursors['entryId'].find('cursor-bottom') >= 0:
                    params['cursor'] = cursors['content']['operation']['cursor']['value']
        
        # if len(msg['timeline']['instructions']) == 1:
        #     cursors = msg['timeline']['instructions'][0]['addEntries']['entries']
        #     for cursor in cursors:
        #         if cursor['entryId'].find('cursor-bottom') >= 0:
        #             # print(cursor['content']['operation']['cursor']['value'])
        #             params['cursor'] = cursor['content']['operation']['cursor']['value']
        # else:
        #     # print(msg['timeline']['instructions'][2].keys())
        #     cursors = msg['timeline']['instructions'][2]['replaceEntry']['entry']
        #     params['cursor'] = cursors['content']['operation']['cursor']['value']
        
        
        
        
        
        for user_time_line in users.keys():
            user = users[user_time_line]
            
            # 获取发帖用户的信息
            user_id_str = user['id_str']
            
            scrapy_user = {
                'user_id': user_time_line,
                'user_name': user['name'],
                'user_screen_name': user['screen_name'],
                'user_description': user['description'],
                'user_friends_count':user['friends_count'],
                'user_follower_count': user['followers_count'],
                'user_favorite_count': user['favourites_count'],
                'user_media_count': user['media_count'],
                'cursor': params['cursor']
            }

            scrapy_user = pd.DataFrame([scrapy_user])
            df = pd.concat([df, scrapy_user], ignore_index=True)


        if i > 0 and i % 10 ==0 :
            df.to_csv(output,index=False)
            print(df)



    df.to_csv(output,index=False)
    print(df)
    return 0

while(1):
    r = main()
    if (r):
        break
    print('切换账号')
 


