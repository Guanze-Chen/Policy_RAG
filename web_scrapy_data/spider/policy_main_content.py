

def extract_DemoQA_pair(url):
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["div"])
    text = docs_transformed[0].page_content
    qa_pattern = r'朋友圈(.*)点击下载'
    qa_matches = re.search(qa_pattern, text)
    qa_content = qa_matches.group(1).strip().replace(" ", '') if qa_matches else ''
    if qa_content == '':
        print(url)
        with open('exception__qa_pair_url.txt', 'a', encoding='utf-8') as f:
            f.write(str(url) + "\n")
        return ''
    text_list = qa_content.split("问：")
    True_QA_pairs = []
    for i in text_list:
        if i != '':
            temp_dict = {}
            try:
                q = i.split("答：")[0]
                a = i.split("答：")[1]
                temp_dict['question'] = q
                temp_dict['answer'] = a
                True_QA_pairs.append(temp_dict)
            except:
                with open('exception__qa_pair_url.txt', 'a', encoding='utf-8') as f:
                    f.write(str(url) + "\n")
                print('---qa文件解析异常')
                print(f'{url}')

    return True_QA_pairs


def extract_title(url):
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["h1"])
    title = docs_transformed[0].page_content.strip()
    return title


def fetch_rag_FileData(urls, base_url='https://www.guang-an.gov.cn'):

    field_list = []
    loader = AsyncChromiumLoader(urls)
    html = loader.load()  # 确保异步加载完成

# 初始化BeautifulSoupTransformer
    bs_transformer = BeautifulSoupTransformer()

    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["div"])

    for i in range(len(urls)):
        docs1 = docs_transformed[i].page_content

        # 提取主题分类
        category_pattern = r"主题分类\s*[:：]?\s*([\u4e00-\u9fa5、]+)"
        category_matches = re.findall(category_pattern, docs1)
        if len(category_matches) == 0:
            print('异常')
            print(urls[i])
            with open('exception__主题_url.txt', 'a', encoding='utf-8') as f:
                f.write(str(urls[i]) + "\n")
            print('++++++++++++++++')
            continue

        # title
        title = extract_title(urls[i])

        # 提取发布日期
        date_pattern = r"发布日期：\s*?\s*(\d{4}-\d{2}-\d{2})"
        date_match = re.search(date_pattern, docs1)
        date = date_match.group(1) if date_match else ''

        # 提取发布机构
        organization_pattern = r"来源\s*[:：]?\s*([\u4e00-\u9fa5]+)"
        organization_match = re.search(organization_pattern, docs1)
        organization = organization_match.group(1).strip() if organization_match else ''

        # 主要内容
        main_pathern = r'朋友圈(.*)二维码'
        main_match = re.search(main_pathern, docs1)
        main_content = main_match.group(1).strip().replace(" ", '') if main_match else ''

        # 使用正则表达式提取“有效性”字段
        validity_pattern = r"有效性\s*[:：]?\s*(\S+)"
        validity_match = re.search(validity_pattern, docs1)
        validity = validity_match.group(1) if validity_match else ''

        # 政策问答字段
        link_pattern = r"政策咨询问答[^\(]*\(([^)]+)\)"
        link_match = re.search(link_pattern, docs1)
        found_links = link_match.group(1) if link_match else ''
        print("qa_link")
        print(found_links)
        if found_links == '':
            with open('empty_qa_link.txt', 'a+', encoding='utf-8') as f:
                f.write(str(urls[i]) + '\n')
        required_field = [category_matches[0], organization, date, main_content, validity]
        if '' in required_field:
            print('以下链接发现必须字段空缺，未提取成功')
            print(urls[i])
            print('------------------')
            with open('exception_url.txt', 'a', encoding='utf-8') as f:
                f.write(str(urls[i]) + "\n")
            continue

        True_QA_pairs = {}
        qa_url = ''
        if found_links != '':
            qa_url = base_url + found_links
            True_QA_pairs = extract_DemoQA_pair(qa_url)
    
        # 写入数据
        url_info_dict = {}

        meta_data_dict = {}
        meta_data_dict['topic'] = category_matches[0]
        meta_data_dict['publish_date'] = date
        meta_data_dict['title'] = title
        meta_data_dict['source'] = organization
        meta_data_dict['url'] = urls[i]
        meta_data_dict['qa_url'] = qa_url
        meta_data_dict['work'] = validity

        url_info_dict['metadata'] = meta_data_dict
        url_info_dict['content'] = main_content
        url_info_dict['T_qa_pairs'] = True_QA_pairs

        field_list.append(url_info_dict)

    return field_list

if __name__ == "__main__":

    import re
    import json
    from langchain_community.document_loaders import AsyncChromiumLoader
    from langchain_community.document_transformers import BeautifulSoupTransformer

    with open("file_url_list.txt","r", encoding="utf-8") as txt_file:
        raw_url_list = txt_file.read()
        urls = eval(raw_url_list)
    chunk_size = 3
    temp_dict = {}
    for i in range(0, len(urls), chunk_size):
        end = i+chunk_size
        chunk_urls = urls[i: end]
        field_list = fetch_rag_FileData(chunk_urls)
        temp_dict['Documents'] = field_list

        with open(f'./data/{i}To{end}.json', 'w', encoding='utf-8') as json_file:
            json_str = json.dumps(temp_dict, ensure_ascii=False)
            json_file.write(json_str)
        print(f'{i}-{end}中的url已经写入-----')

        
        
        

    

    



    



    

