# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: sohu_auto.py 
@Time: 2018/9/28 16:59
@Software: PyCharm 
@Description:
"""
import hashlib
import requests
import re
from lxml import etree
import pandas as pd
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
}


def get_car_urls():

    url='http://db.auto.sohu.com/home/iframe.html'
    res=requests.get(url,headers=headers)
    url_pattern='<li>.*?<a.*?href="(.*?)".*?</a>.*?</li>'
    urls=re.findall(url_pattern,res.text,re.I|re.S|re.M)
    dianping_urls=['http:'+url+'/dianping.html' for url in urls if 'brand' not in url]
    return dianping_urls


dianping_urls=get_car_urls()


def get_car_commnet():
    content_id=[]
    sentiment_value = []
    pos_content = []
    neg_content = []

    url='http://db.auto.sohu.com/haval-2128/2834/dianping.html'
    res = requests.get(url, headers=headers)
    html = etree.HTML(res.text)
    print(res.text)
    koubei_tabcons = html.xpath('//div[@class="koubei-tabcon "]')[1:3]
    print(len(koubei_tabcons))
    pos_koubei, neg_koubei = koubei_tabcons[0], koubei_tabcons[1]
    pos_comm = [comm.strip() for comm in
                pos_koubei.xpath('//ul/li/div[@class="comm-content"]/p[@class="short-comm"]/text()')]
    neg_comm = [comm.strip() for comm in
                neg_koubei.xpath('//ul/li/div[@class="comm-content"]/p[@class="short-comm"]/text()')]
    pos_content.extend(pos_comm)
    neg_content.extend(neg_comm)


    for _ in pos_content:
        sentiment_value.append(1)
    for _ in neg_content:
        sentiment_value.append(-1)

    for _,con in zip(sentiment_value,pos_content+neg_content):
        md = hashlib.md5(con.encode(encoding="utf-8")).hexdigest()[8:-8]
        content_id.append(md)
    data={
        'content_id':content_id,
        'content':pos_content+neg_content,
        'subject':None,
        'sentiment_value':sentiment_value,
        'sentiment_word':None
    }
    df=pd.DataFrame(data)

    df.to_csv('sohu_auto.csv',index=False,columns=['content_id','content','subject','sentiment_value','sentiment_word'])


get_car_commnet()