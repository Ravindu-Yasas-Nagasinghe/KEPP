import json

#train data
with open('/l/users/ravindu.nagasinghe/New_STEP/PDPP/outputs/train_data_list.json', 'r') as original_data_file:
    original_data = json.load(original_data_file)

# Load the large list from a text file
with open('/l/users/ravindu.nagasinghe/New_STEP/PDPP/final_list_T4_train.json', 'r') as large_list_file:
    large_list = json.load(large_list_file)


for i, item in enumerate(original_data):
    item["id"]["pred_list"] = large_list[i]

with open('/l/users/ravindu.nagasinghe/New_STEP/PDPP/final_list_T4_train_final.json', 'w') as modified_data_file:
    json.dump(original_data, modified_data_file)

#test data

with open('/l/users/ravindu.nagasinghe/New_STEP/PDPP/outputs/test_data_list.json', 'r') as original_data_file:
    original_data = json.load(original_data_file)

# Load the large list from a text file
with open('/l/users/ravindu.nagasinghe/New_STEP/PDPP/final_list_T4_test.json', 'r') as large_list_file:
    large_list = json.load(large_list_file)


for i, item in enumerate(original_data):
    item["id"]["pred_list"] = large_list[i]

with open('/l/users/ravindu.nagasinghe/New_STEP/PDPP/final_list_T4_test_final.json', 'w') as modified_data_file:
    json.dump(original_data, modified_data_file)