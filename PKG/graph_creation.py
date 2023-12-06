import os
import pickle
import networkx as nx
import logging
import json
import torch
from tabulate import tabulate
from action_dictionary import action_dictionary
import numpy as np
import random
import cv2
# Define the path to save/load the trained graph and checkpoints
graph_save_path = '/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/graphs/out_edge_N_graph/trained_graph.pkl'

def get_all_paths(graph, start_node, end_node, cutoff):
  paths = []
  def _dfs(node, path, cutoff):
    if node == end_node:
      paths.append(path + [node])
      #return

    if cutoff is not None and len(path) >= cutoff:
      return

    for neighbor in graph.neighbors(node):
      #if neighbor not in path:
      _dfs(neighbor, path + [node], cutoff)
    
    # Add a condition to output paths that loop back to the end node.
    #if node == end_node and path[-1] == end_node:
    #  paths.append(path + [node])

  _dfs(start_node, [], cutoff)
  return paths


def train_graph_output_edge_normalize(video_sequences):
    # Step 1: Load the graph if it exists; otherwise, create a new graph
    if os.path.exists(graph_save_path):
        # Load the trained graph
        with open(graph_save_path, 'rb') as graph_file:
            graph = pickle.load(graph_file)
    else:
        # Create a new graph and add edges with accumulated weights
        graph = nx.DiGraph()
        iteration = 1
        for seq in video_sequences.values():
            for i in range(len(seq) - 1):
                current_node, next_node = seq[i], seq[i + 1]
                if graph.has_edge(current_node, next_node):
                    graph[current_node][next_node]['weight'] += 1
                else:
                    graph.add_edge(current_node, next_node, weight=1)
            iteration = iteration+1

        # Normalize the graph as described
        for node in graph.nodes:
            # Calculate the sum of edge weights for outgoing edges from the current node
            outgoing_sum = sum(graph[node][neighbor]['weight'] for neighbor in graph.successors(node))
            
            # Normalize the weights of outgoing edges (including self-loops)
            for neighbor in graph.successors(node):
                graph[node][neighbor]['weight'] /= outgoing_sum


        print('graph completed')

         # Save the trained graph
        with open(graph_save_path, 'wb') as graph_file:
            pickle.dump(graph, graph_file)

        # Step 4: Add logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    logging.info("Graph training completed.")

    # Step 3: Validate the function with input and output nodes
    start_node = 2
    end_node = 4
    n = 3  #numbere of paths required
    top_paths = find_top_n_paths(graph, start_node, end_node, n, 6)



    print(f"Highest weight paths from {start_node} to {end_node} are:")
    for i, (path, weight) in enumerate(top_paths, 1):
        print(f"[Path {i}] Weight: {weight}, Path: {path}")
    print(top_paths[0])


def train_graph_min_max_normalize(video_sequences):
    # Step 1: Load the graph if it exists; otherwise, create a new graph
    if os.path.exists(graph_save_path):
        # Load the trained graph
        with open(graph_save_path, 'rb') as graph_file:
            graph = pickle.load(graph_file)
    else:
        # Create a new graph and add edges with accumulated weights
        graph = nx.DiGraph()
        iteration = 1
        for seq in video_sequences.values():
            for i in range(len(seq) - 1):
                current_node, next_node = seq[i], seq[i + 1]
                if graph.has_edge(current_node, next_node):
                    graph[current_node][next_node]['weight'] += 1
                else:
                    graph.add_edge(current_node, next_node, weight=1)
            iteration = iteration+1

        print('graph completed')

        # Find the maximum and minimum weights in the graph
        max_weight = max((d['weight'] for _, _, d in graph.edges(data=True)))
        min_weight = min((d['weight'] for _, _, d in graph.edges(data=True)))
        print(min_weight)
        print(max_weight)

        # Normalize the weights in the graph using min-max normalization
        for u, v, d in graph.edges(data=True):
            d['weight'] = (d['weight'] - min_weight) / (max_weight - min_weight)


        # Save the trained graph
        with open(graph_save_path, 'wb') as graph_file:
            pickle.dump(graph, graph_file)

        # Step 4: Add logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    logging.info("Graph training completed.")

    # Step 3: Validate the function with input and output nodes
    start_node = 2
    end_node = 4
    n = 3  #numbere of paths required
    top_paths = find_top_n_paths(graph, start_node, end_node, n, 4)



    print(f"Highest weight paths from {start_node} to {end_node} are:")
    for i, (path, weight) in enumerate(top_paths, 1):
        print(f"[Path {i}] Weight: {weight}, Path: {path}")
    print(top_paths[0])


def calculate_weight_sequence_method_one(graph, path):
    weight_sequence = []
    edge_counts = {}  # Dictionary to keep track of the number of times each edge is traversed

    for i in range(len(path) - 1):
        source_node = path[i]
        target_node = path[i + 1]
        edge_key = (source_node, target_node)

        # Check if the edge has been traversed before
        if edge_key in edge_counts:
            n = edge_counts[edge_key]  # Get the number of times the edge has been traversed
            edge_counts[edge_key] += 1  # Increment the count
            weight = graph[source_node][target_node]['weight'] / n
        else:
            # If the edge is traversed for the first time, set n=1 and store it in edge_counts
            edge_counts[edge_key] = 2
            weight = graph[source_node][target_node]['weight']

        weight_sequence.append(weight)

    return weight_sequence

def calculate_weight_sequence_method_two(graph, path):
    weight_sequence = []
    edge_counts = {}  # Dictionary to keep track of the number of times each edge is traversed

    for i in range(len(path) - 1):
        source_node = path[i]
        target_node = path[i + 1]
        edge_key = (source_node, target_node)

        # Check if the edge has been traversed before
        if edge_key in edge_counts:
            n = edge_counts[edge_key]  # Get the number of times the edge has been traversed
            edge_counts[edge_key] += 1  # Increment the count
    
        else:
            # If the edge is traversed for the first time, set n=1 and store it in edge_counts
            edge_counts[edge_key] = 1
            weight = graph[source_node][target_node]['weight']
    for i in range(len(path) - 1):
        source_node = path[i]
        target_node = path[i + 1]
        edge_key = (source_node, target_node)
        n = edge_counts[edge_key]
        weight = ((graph[source_node][target_node]['weight'])**n)/n

        weight_sequence.append(weight)
    return weight_sequence


# Step 2: Implement a function to find the top 3 highest-weighted paths
def find_top_n_paths(graph, start_node, end_node, n, max_path_length):

    if not graph.has_node(start_node) or not graph.has_node(end_node):
        return []  # Skip if start or end node is not in the graph

    #all_paths = list(nx.all_simple_paths(graph, start_node, end_node, cutoff=max_path_length))
    # Get all paths, including loops.
    all_paths = get_all_paths(graph, start_node, end_node,max_path_length)

    all_paths_with_weights = []
    weight_sequences = []
    for path in all_paths:
        if len(path)==max_path_length:

            #weight = sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))          #For method 0  (default weightage)
            #weight_sequence = [graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)]

            #weight_sequence = calculate_weight_sequence_method_one(graph, path)   #for method 1
            #weight = sum(weight_sequence)

            #weight_sequence = calculate_weight_sequence_method_two(graph, path)   #for method 2 
            #weight = sum(weight_sequence)

            weight_sequence = [graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)]      #for method 3
            weight = np.prod(weight_sequence)


            all_paths_with_weights.append((path, weight))
            weight_sequences.append((path, weight_sequence, weight))  # Append the weight sequence
        else:
            continue

    sorted_paths = sorted(all_paths_with_weights, key=lambda x: x[1], reverse=True)
    weight_sequences_paths = sorted(weight_sequences, key=lambda x: x[2], reverse=True)
    if mode == 'validate' or mode == 'train_minmax' or mode == 'train_out_n':
        if len(sorted_paths) >= n:
            top_n_paths = sorted_paths[:n]
        else:
            top_n_paths = sorted_paths
        return top_n_paths
    elif mode == 'seq_view':
        if len(weight_sequences_paths) >= n:
            top_n_paths = weight_sequences_paths[:n]
        else:
            top_n_paths = weight_sequences_paths
        return top_n_paths

def are_lists_similar(list1, list2):
    # Check if the lists have the same length
    if len(list1) != len(list2):
        return False
    
    # Compare elements of the lists
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return False
    
    # If all elements are equal, the lists are similar
    return True

# Perform validation using the graph
def validate_graph(videos,testing_sequences):
    # Load the trained graph 
    with open(graph_save_path, 'rb') as graph_file:
        graph = pickle.load(graph_file)
    path_Set =[]
    vid_number  = 0
    for seq in testing_sequences:
        start = seq[0]
        end = seq[1]
        paths = find_top_n_paths(graph, start, end,n=1, max_path_length=6)
        #if paths:
        path_Set.append([item[0] for item in paths])

        print('vid:', vids[vid_number] , paths)
        vid_number = vid_number+1

    print('len path set:', len(path_Set))
    differing_sublists ={}
    for i in range(len(videos)):
        #all_sublists_differ = all(not torch.equal(sublist, video_sequences[i]) for sublist in path_Set[i])
        flag = False
        for sublist in path_Set[i]:
            flag = are_lists_similar(videos[i], sublist) 
            if flag:
                break

        if not flag:
            differing_sublists = {
                'i_batch': i,
                'video_sequences_list': videos[i],
                'predicted_lists': [sublist for sublist in path_Set[i]],
            }

        print(differing_sublists)

def validate_graph_final(testing_sequences, graph_):
    # Load the trained graph 

    path_Set =[]

    
    start = testing_sequences[0]
    end = testing_sequences[1]
    paths = find_top_n_paths(graph_, start, end,n=1, max_path_length=6)
        #if paths:
    path_Set.append([item[0] for item in paths])
    '''
    if (len(path_Set[0])==0):
        path_Set[0] = [[start,start,start, end, end, end], [start,start,start, end, end, end]]  ###zero pad if path does not exist in knowledge graph
        print(path_Set)
    if (len(path_Set[0])==1):
        path_Set[0] = [path_Set[0], path_Set[0]]  ###zero pad if path does not exist in knowledge graph
        print(path_Set)
    '''
    if (len(path_Set[0])==0):
        path_Set[0] = [[start,start,start, end, end, end]]    ###zero pad if path does not exist in knowledge graph
        print(path_Set)
    return path_Set

def sequence_viewing(testing_sequences, max_path_length,n):
    with open(graph_save_path, 'rb') as graph_file:
        graph = pickle.load(graph_file)

    start = testing_sequences[0]
    end = testing_sequences[1]
    
    paths = find_top_n_paths(graph, start, end,n, max_path_length)

    result_list = []

    for sublist in paths:
        value_sequence = [action_dictionary.get(num+1, "Not Found") for num in sublist[0]]
        result_list.append([value_sequence,sublist[1],sublist[2]])

        print('actions: ', value_sequence , 'weights: ', sublist[1], 'Tot_weight', sublist[2])

    headers = ["Actions", "Weights", "Total Weight"]
    table = tabulate(result_list, headers, tablefmt="fancy_grid")
    print(table)
# Define a function to find neighbors and their weights
def find_all_direct_connections(graph, node, depth = 2):
    direct_connections = []
    

    if node in graph and depth > 0:
        neighbors = list(graph.neighbors(node))
        #neighbors.remove(node)
        print('Neighbors to node: ', node,' are: ',neighbors)
        action_from = action_dictionary.get(node+1, f'Action {node}')
        for neighbor in neighbors:
            action_to = action_dictionary.get(neighbor+1, f'Action {neighbor}')


            # Check if there is a direct connection from input node to neighbor
            if graph.has_edge(node, neighbor):
                edge_weight = graph[node][neighbor]['weight']
                connection = (node, neighbor, '->', edge_weight)
                if connection not in visited_connections:
                    direct_connections.append({
                        'From Node': node,
                        'To Node': neighbor,
                        'Direction': '->',  # Indicates the direction of the edge from input node to neighbor
                        'Weight': edge_weight,
                        'Action From': action_from,
                        'Action To': action_to
                    })
                    visited_connections.add(connection)
            
            # Check if there is a direct connection from neighbor to input node
            if graph.has_edge(neighbor, node):
                edge_weight = graph[neighbor][node]['weight']
                connection = (neighbor, node, '->', edge_weight)
                if connection not in visited_connections:
                    direct_connections.append({
                        'From Node': neighbor,
                        'To Node': node,
                        'Direction': '->',  # Indicates the direction of the edge from neighbor to input node
                        'Weight': edge_weight,
                        'Action From': action_to,
                        'Action To': action_from
                    })
                    visited_connections.add(connection)

            nested_connections = find_all_direct_connections(graph, neighbor, depth=depth -1)
            direct_connections.extend(nested_connections)

        # Check for connections from input node to itself (self-loop)
        if graph.has_edge(node, node):
            edge_weight = graph[node][node]['weight']
            connection = (node, node, '->', edge_weight)
            if connection not in visited_connections:
                direct_connections.append({
                    'From Node': node,
                    'To Node': node,
                    'Direction': '->',  # Indicates the direction of the self-loop
                    'Weight': edge_weight,
                    'Action From': action_from,
                    'Action To': action_from
                })
                visited_connections.add(connection)

    visualize_image(direct_connections)
    return direct_connections

# Function to calculate control points for Bézier curve
def calculate_control_points(p1, p2):
    # Midpoint between p1 and p2
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    # Offset from mid to control points
    offset = (int((random.random() - 0.5) * 200), int((random.random() - 0.5) * 200))
    control1 = (mid[0] + offset[0], mid[1] + offset[1])
    control2 = (mid[0] - offset[0], mid[1] - offset[1])
    return control1, control2

def visualize_image(direct_connections):
    all_nodes = set()
    for connection in direct_connections:
        all_nodes.add(connection['From Node'])
        all_nodes.add(connection['To Node'])

# Create a dictionary to store node positions with random coordinates
    node_positions = {}
    #i = 0
    #x = 100
    #y = 100
    for node in all_nodes:
        x = random.randint(50, 950)  # Random X-coordinate between 50 and 750
        y = random.randint(50, 950)  # Random Y-coordinate between 50 and 550
        node_positions[node] = (x, y)

    # Create a blank image
    image_width = 1000
    image_height = 1000
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image.fill(255)  # White background

    # Draw nodes as red circles
    for node, pos in node_positions.items():
        cv2.circle(image, pos, 20, (0, 0, 255), -1)  # Red circle

    # Draw edges with weights and actions
    for connection in direct_connections:
        from_node = connection['From Node']
        to_node = connection['To Node']
        weight = connection['Weight']
        action_from = connection['Action From']
        action_to = connection['Action To']

        from_pos = node_positions[from_node]
        to_pos = node_positions[to_node]

        # Draw the edge from "From Node" to "To Node"
        is_reverse = from_pos[0] >= to_pos[0]

        #cv2.arrowedLine(image, from_pos, to_pos, (255, 0, 0), 2, tipLength=0.01)  # Arrowed line
        if not is_reverse:
            cv2.arrowedLine(image, from_pos, to_pos, (0, 150, 0), 2, tipLength=0.01)  # Arrowed line
        else:
            cv2.arrowedLine(image, from_pos, to_pos, (255, 0, 0), 2, tipLength=0.02, shift=0, line_type=cv2.LINE_8)  # Arrowed line for reverse direction

        # Calculate text position for weight
        text_x = (from_pos[0] + to_pos[0]) // 2
        text_y = (from_pos[1] + to_pos[1]) // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(str(weight), font, font_scale, font_thickness)
        text_x -= text_size[0] // 2
        text_y += text_size[1] // 2

        # Draw weight text above the edge
        cv2.putText(image, str(weight), (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

        # Draw action labels near the nodes
        cv2.putText(image, action_from, (from_pos[0] - 30, from_pos[1] - 30), font, 0.5, (0, 0, 0), 1)
        cv2.putText(image, action_to, (to_pos[0] - 30, to_pos[1] - 30), font, 0.5, (0, 0, 0), 1)

    # Display the image (you can save it using cv2.imwrite)
    file_path = "graph_visualization.png"
    cv2.imwrite(file_path, image)



if __name__ == "__main__":

    video_sequences = {}

    mode = input("Select mode (train_minmax or train_out_n or validate or visualize or seq_view): ").lower()
    if mode == 'train_minmax':
        with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/training_action_list.txt', 'r') as file:
            for line in file:
                # Parse each line as a JSON object
                sequence_data = json.loads(line)
                vid = sequence_data['vid']
                legal_range = sequence_data['legal_range']
                
                # Store only the legal_range part
                video_sequences[vid] = legal_range

        print(video_sequences)
        train_graph_min_max_normalize(video_sequences)

    elif mode == 'train_out_n':
        #with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/data/training_action_list.txt', 'r') as file:
        #with open('/l/users/ravindu.nagasinghe/MAIN_codes/CrossTask_base/temp/PDPP/training_action_list_Cross_base.txt', 'r') as file:
        #with open('/l/users/ravindu.nagasinghe/MAIN_codes/NIV/step/PDPP/outputs/testing_action_list.txt', 'r') as file:
        with open('/home/ravindu.nagasinghe/GithubCodes/COIN/step/PDPP/train_action_list_coin.txt', 'r') as file:
            for line in file:
                # Parse each line as a JSON object
                sequence_data = json.loads(line)
                vid = sequence_data['vid']
                legal_range = sequence_data['legal_range']
                
                # Store only the legal_range part
                video_sequences[vid] = legal_range

        print(video_sequences)
        train_graph_output_edge_normalize(video_sequences)
    elif mode == 'validate':
        testing_sequence =[]
        videos=[]
        vids =[]
        with open('/home/ravindu.nagasinghe/GithubCodes/PDPP/PDPP/final_list_step_test_MODEL.json', 'r') as file:
            data = json.load(file)
            with open(graph_save_path, 'rb') as graph_file:
                graph_ = pickle.load(graph_file)
            for item in data:
                # Parse each line as a JSON object
                sequence_data = item['id']
                vid = sequence_data['vid']
                legal_range = sequence_data['legal_range']
                pred_list = sequence_data['pred_list']
                # Store only the legal_range part
                #video_sequences[vid] = legal_range
                videos.append(legal_range)
                #vids.append(vid)

                #first_digit = legal_range[0][-1]
                #last_digit = legal_range[-1][-1]
                first_digit = pred_list[0]
                last_digit = pred_list[-1]

                # Create a list containing the first and last digits
                result_list = [first_digit, last_digit]
                path = validate_graph_final(result_list, graph_)
                
                sequence_data['graph_action_path'] = path[0][0]  #####for n=1
                #sequence_data['graph_action_path'] = path[0]  #####for n>1

                #testing_sequence.append(result_list)
        with open('/home/ravindu.nagasinghe/GithubCodes/RaviPP/final_list_step_test_MODEL_PKG.json', 'w') as outfile:
            json.dump(data, outfile)
        #print(video_sequences)
        #print(testing_sequence)
        #validate_graph(videos, testing_sequence)
    elif mode == 'visualize':
        with open(graph_save_path, 'rb') as graph_file:
            graph = pickle.load(graph_file)
        # Input node
        input_node = int(input("Enter a value for input_node: "))  #  input node
        print('visualize connections to node ......', input_node, ' Which is ',action_dictionary.get(input_node+1, f'Action {input_node}'))
        # Find neighbors and their weights
        visited_connections = set() 
        neighbors_data = find_all_direct_connections(graph, input_node, depth=2)

        # Display results in a table
        if neighbors_data:
            table_headers = ['From Node', 'Direction', 'To Node', 'Weight', 'Action From','Direction', 'Action To']
            table_data = [(d['From Node'], d['Direction'], d['To Node'], d['Weight'], d['Action From'], d['Direction'], d['Action To']) for d in neighbors_data]
            print(tabulate(table_data, headers=table_headers, tablefmt='pretty'))
        else:
            print(f"No direct connections found for node {input_node}.")

    elif mode == 'seq_view':
        # Input node
        max_path_length, n =  map(int, input("Enter max_path length and number of sequences required ").split())
        max_path_length = max_path_length - 1 # because we give the intermediate number of steps as input to the code. not number of action steps.
        try:
            num1, num2  = map(int, input('Enter start, end action numbers').split())
            testing_sequences = [num1, num2]
        except:
            print("Invalid input. Please enter two space-separated numbers.")
        
        sequences = sequence_viewing(testing_sequences, max_path_length,n)

    else:
        print("Invalid mode. Use 'train' or 'validate'.")
