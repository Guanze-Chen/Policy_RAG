import requests
from scrapy import Selector
import json
import time
import random

def fetch_file_url_list(total_page=7):
    """
    The function to fetch polciy url

    return: url_list []
    """
    res_list = []
    url = "https://www.guang-an.gov.cn/queryList"
    base_url = "https://www.guang-an.gov.cn"

    for page_num in range(1, total_page):
        data = {
            "current": page_num,
            "pageSize": 15,
            "webSiteCode[]": "gasrmzfw",
            "channelCode[]": "c104243"
        }
        
        time.sleep(0.2 + random.uniform(0.5, 1.2))

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36"}
        res = requests.post(url,headers=headers, data=data)
        res = res.json()

        target = res['data']['results']
        for item in target:
            # print(item)
            target = item['source']
            target_url = eval(target['urls'])
            real_url = base_url + target_url['pc']
            res_list.append(real_url)
    
    return res_list



if __name__ == "__main__":
    
    file_url_list = fetch_file_url_list(total_page=7)
    with open("file_url_list.txt", "w", encoding='utf-8') as txt_file:
        txt_file.write(str(file_url_list))