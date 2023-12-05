import json
'''
# Load the JSON data from a file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/final_output_test_data_list.json', 'r') as data_file:
    data = json.load(data_file)

# Iterate through the data and add "LLM_action_path" key
for item in data:    
    item["id"]["LLM_action_path"] = item["id"]["graph_action_path"]

# Write the modified data back to a JSON file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/final_output_test_data_list_2.json', 'w') as modified_data_file:
    json.dump(data, modified_data_file)

print("Modified data has been written to 'modified_data.json'.")

# Load the JSON data from a file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/Training_PKG_list.json', 'r') as data_file:
    data = json.load(data_file)

# Iterate through the data and add "LLM_action_path" key
for item in data:    
    item["id"]["LLM_action_path"] = item["id"]["graph_action_path"]

# Write the modified data back to a JSON file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/Training_PKG_LLM_list.json', 'w') as modified_data_file:
    json.dump(data, modified_data_file)
'''
####################################

# Load the JSON data from a file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/Training_PKG_list_T3_window.json', 'r') as data_file:
    data = json.load(data_file)

# Iterate through the data and add "LLM_action_path" key
for item in data:    
    item["id"]["LLM_action_path"] = item["id"]["graph_action_path"]

# Write the modified data back to a JSON file
with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/outputs/Training_PKG_LLM_list_T3_window.json', 'w') as modified_data_file:
    json.dump(data, modified_data_file)

print("Modified data has been written to 'modified_data.json'.")
