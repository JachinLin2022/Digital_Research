# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import pandas as pd
import os
import json
import time
import asyncio
import aiohttp
import threading

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
key = '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'
openai.api_key = key


PROMPT = '''
<{}>
Does this sentence in <> mention Bitcoin?
'''


def chat(messages):
    # print(messages)
    while (1):
        try:
            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            break
        except Exception as e:
            print(e)
    result = rsp['choices'][0]['message']
    return result


def step_chat(input):
    messages = [
        {"role": "user", "content": "I will ask you questions. Please only answer with yes, no or unsure."},
        {"role": "assistant", "content": "Okay, I'm ready."},
    ]
    messages.append({"role": "user", "content": PROMPT.format(input)})

    answer_message = chat(messages)
    messages.append(answer_message)

    if answer_message['content'].lower().find('yes') >= 0:
        format = ' Answer with [yes,no,unsure]'
        # format = ''
        messages.append(
            {"role": "user", "content": "Is the sentence in <> a statement?" + format})
        answer_message = chat(messages)
        messages.append(answer_message)

        if answer_message['content'].lower().find('yes') >= 0:

            statement_questions = [
                'Is it news about bitcoin?' + format,
                'Is it a cryptocurrency advertisement?' + format,
                'Is it an objective evaluation or opinion of Bitcoin?' + format,
                "Is it an analysis or forecast of Bitcoin's trend?" + format
            ]

            # if answer_message['content'].lower().find('yes') >= 0:
            for question in statement_questions:
                message = {
                    "role": "user",
                    "content": question
                }
                messages.append(message)
                answer_message = chat(messages)
                messages.append(answer_message)

        elif answer_message['content'].lower().find('no') >= 0:
            print(answer_message['content'].lower())
            not_statement_questions = [
                'Does it expresse a personal subjective question?' + format,
                'Is it a rhetorical question regarding Bitcoin news?' + format,
                'Is it a rhetorical question about the trend of Bitcoin?' + format
            ]

            # if answer_message['content'].lower().find('no') >= 0:
            for question in not_statement_questions:
                message = {
                    "role": "user",
                    "content": question
                }
                messages.append(message)
                answer_message = chat(messages)
                messages.append(answer_message)
    return messages

def classify(history):
    messages = json.loads(history)
    prompt = '''
    Now you are a Bitcoin text filter, and your goal is to filter out texts that are not related to Bitcoin price fluctuations.
    Please help me determine whether the sentence in <> given before may affect Bitcoin price.
    Analyze step by step based on the above questions.
    Output in json format and key is label. If It is related, output 1. If not, output 0.
    '''
    messages.append({
        "role": "user",
        "content":prompt
    })
    
    answer_message = chat(messages)
    return answer_message
    # print(answer_message)
    
    
    
    
tweets = pd.read_csv('steps_result.csv')

# tweets['steps_result'] = ''
# tweets['analysis'] = ''

for idx, row in tweets.iterrows():

    print(idx)
    input = row['full_text']
    history = row['steps_result']
    messages = step_chat(input)
    # answer_message = classify(history)

    
    
    # tweets.at[idx, 'analysis'] = answer_message.content
    tweets.at[idx, 'steps_result'] = json.dumps(messages,indent=4)
    if (idx+1) % 1 == 0:
        print('save')
        tweets.to_csv('steps_result2.csv', index=False)
        # tweets[['full_text', 'steps_result']].to_csv(
        #     'steps_result.csv', index=False)

    # for message in messages:
    #     print(message)
    # break
# print(messages)
