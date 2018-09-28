# !/usr/bin/env python
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: demo.py 
@Time: 2018/9/28 17:09
@Software: PyCharm 
@Description:
"""

# coding=utf-8
import re

content = '''
<div class="ph_item">
                        <strong>
                        小型车
                        </strong>
                        <ol class="toplist">
                            
                                <li>
                                <em class="on">1</em>
                                <a target="_blank" href="//db.auto.sohu.com/dongfenghonda/1885" class="cname">本田XR-V</a>
                                <span class="red">12.78万起</span>
                                </li>
                            
                                <li>
                                <em class="on">2</em>
                                <a target="_blank" href="//db.auto.sohu.com/geelyauto/5345" class="cname">吉利远景X3</a>
                                <span class="red">5.09万起</span>
                                </li>
                            
                                <li>
                                <em class="on">3</em>
                                <a target="_blank" href="//db.auto.sohu.com/dongfengnissan/5292" class="cname">日产劲客</a>
                                <span class="red">9.98万起</span>
                                </li>
                            
                                <li>
                                <em>4</em>
                                <a target="_blank" href="//db.auto.sohu.com/guangqitoyota/5460" class="cname">丰田C-HR</a>
                                <span class="red">14.48万起</span>
                                </li>
                            
                                <li>
                                <em>5</em>
                                <a target="_blank" href="//db.auto.sohu.com/guangqihonda/4446" class="cname">本田缤智</a>
                                <span class="red">12.88万起</span>
                                </li>
                            
                                <li>
                                <em>6</em>
                                <a target="_blank" href="//db.auto.sohu.com/shanghaivw/2210" class="cname">大众Polo</a>
                                <span class="red">7.99万起</span>
                                </li>
                            
                                <li>
                                <em>7</em>
                                <a target="_blank" href="//db.auto.sohu.com/guangqihonda/2234" class="cname">本田飞度</a>
                                <span class="red">7.38万起</span>
                                </li>
                            
                                <li>
                                <em>8</em>
                                <a target="_blank" href="//db.auto.sohu.com/baojun-2088/5174" class="cname">宝骏510</a>
                                <span class="red">5.48万起</span>
                                </li>
                            
                                <li>
                                <em>9</em>
                                <a target="_blank" href="//db.auto.sohu.com/yiqitoyota/1012" class="cname">丰田威驰</a>
                                <span class="red">6.98万起</span>
                                </li>
                            
                                <li>
                                <em>10</em>
                                <a target="_blank" href="//db.auto.sohu.com/beijinghyundai/2903" class="cname">现代瑞纳三厢</a>
                                <span class="red">4.99万起</span>
                                </li>
                            
                        </ol>
                    </div>
'''

# 获取<a href></a>之间的内容
print('获取链接文本内容:')
con_pattern = '<a.*?>(.*?)</a>'
con=re.findall(con_pattern,content,re.S|re.M)
print(con)

print("获取链接内容")
url_pattern='<li>.*?<a.*?href="(.*?)".*?</a>.*?</li>'
urls=re.findall(url_pattern,content,re.I|re.S|re.M)
print(urls)