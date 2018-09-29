# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: sina_auto.py 
@Time: 2018/9/29 9:47
@Software: PyCharm 
@Description:
"""
import hashlib
import requests
from lxml import etree
import pandas as pd
import re
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
}


def get_car_urls():
    url='http://db.auto.sina.com.cn/'
    res=requests.get(url=url,headers=headers)
    html=etree.HTML(res.text)
    urls=html.xpath('//div[@class="y-tuku235 seek-list"]/ul/li/div/a/@href')
    prefix=' http://data.auto.sina.com.cn/car_comment/'
    car_urls=[prefix+u.split('/')[-2] for u in urls]
    return car_urls


car_urls=get_car_urls()

def get_car_comment():
    content_id = []
    sentiment_value = []
    pos_content = []
    neg_content = []
    for url in car_urls:
        print(url)
        # url='http://data.auto.sina.com.cn/car_comment/1933'
        res=requests.get(url,headers=headers)
        res.encoding='gbk'

        pos_pattern = re.compile(r'<p class="yo"><strong class="o">优点：</strong>(.*?)</p>', re.I | re.S | re.M)
        pos = [pos.strip() for pos in re.findall(pos_pattern, res.text)]

        neg_pattern=re.compile(r'<p class="qu"><strong class="q1">缺点：</strong>(.*?)</p>',re.I|re.S|re.M)
        neg=[pos.strip() for pos in re.findall(neg_pattern,res.text)]

        pos_content.extend(pos)
        neg_content.extend(neg)

    for _ in pos_content:
        sentiment_value.append(1)
    for _ in neg_content:
        sentiment_value.append(-1)

    for _,con in zip(sentiment_value,pos_content+neg_content):
        md = hashlib.md5(con.encode(encoding="utf-8")).hexdigest()[8:-8]
        content_id.append(md)

    data = {
        'content_id': content_id,
        'content': pos_content + neg_content,
        'subject': None,
        'sentiment_value': sentiment_value,
        'sentiment_word': None
    }
    df = pd.DataFrame(data)
    df.to_csv('sina_auto.csv', index=False,
              columns=['content_id', 'content', 'subject', 'sentiment_value', 'sentiment_word'])

get_car_comment()