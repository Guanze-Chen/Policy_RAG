import json
import os

data_path = r'./data'

data_files = os.listdir(data_path)
work_policy_list = []
temp_dict = {}
for file_name in data_files:
    file_path = data_path + '\\' + file_name
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    core_data = data['Documents']
    for policy in core_data:
        work = policy['metadata']['work']
        if work == '有效':
            work_policy_list.append(policy)

temp_dict['data'] = work_policy_list


with open('work_policy_data.json', 'w', encoding='utf-8') as json_file:
    json_str = json.dumps(temp_dict, ensure_ascii=False)
    json_file.write(json_str)

