import json

# Load the JSON data from a file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/Testing_PKG_LLM_list.json', 'r') as data_file:
    data = json.load(data_file)
for item in data:
    if len(item["id"]["legal_range"]) != 4 or len(item["id"]["graph_action_path"]) != 4 or len(item["id"]["LLM_action_path"]) != 4:
        print(item['id'])
        print('Error!!!!!')
print('finished')