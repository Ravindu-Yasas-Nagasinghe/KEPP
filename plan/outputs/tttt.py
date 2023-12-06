import json

# Specify the path to your JSON file
json_file_path = '/l/users/ravindu.nagasinghe/T4How/multiple_plans/PDPP/outputs/test_list.json'

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Check if the key "play" exists in the dictionary
for ids in data:
    print(ids)
    list1 = ids["graph_action_path"]

    # Find the length of the list
    play_list_length = len(list1)
    if play_list_length !=2:
        print(f"The length of the 'play' list is: {play_list_length}")
print('Finish')