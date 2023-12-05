import networkx as nx
import cv2
import numpy as np

import os
import pickle
import networkx as nx
import logging
import json
import torch
from tabulate import tabulate
from action_dictionary import action_dictionary

# Function to find all paths from an input node within a fixed length
def find_paths_within_length(graph, start_node, max_length):
    def dfs_paths(node, path, length):
        if length <= max_length:
            path.append(node)
            if node == start_node:
                yield list(path)
            else:
                for neighbor in graph.neighbors(node):
                    if neighbor not in path:
                        yield from dfs_paths(neighbor, path, length + 1)
            path.pop()

    paths = list(dfs_paths(start_node, [], 0))
    return paths


graph_save_path = '/home/ravindu.nagasinghe/GithubCodes/RaviPP/trained_graph.pkl'

with open(graph_save_path, 'rb') as graph_file:
    graph = pickle.load(graph_file)

# Input parameters
input_node = 1  # Replace with the desired input node
max_length = 4  # Replace with the desired fixed length

# Find all paths within the fixed length from the input node
all_paths = find_paths_within_length(graph, input_node, max_length)

# Create an image for visualization
node_positions = nx.circular_layout(graph)
image_width = len(all_paths[0]) * 100  # Width of the image based on the number of steps
image_height = len(graph.nodes()) * 100  # Height of the image based on the number of nodes

image = np.zeros((image_height, image_width, 3), dtype=np.uint8)  # Initialize an empty image
image.fill(255)  # Set the background to white

# Draw nodes
for node, pos in node_positions.items():
    x, y = int(pos[0] * image_width), int(pos[1] * image_height)
    cv2.circle(image, (x, y), 10, (0, 0, 0), -1)  # Draw black nodes

# Draw paths with thickness based on weight
for path in all_paths:
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        weight = graph[u][v]['weight']
        thickness = int((weight / 5) * 3)  # Adjust the scaling factor as needed
        u_pos, v_pos = node_positions[u], node_positions[v]
        u_x, u_y = int(u_pos[0] * image_width), int(u_pos[1] * image_height)
        v_x, v_y = int(v_pos[0] * image_width), int(v_pos[1] * image_height)
        cv2.line(image, (u_x, u_y), (v_x, v_y), (0, 0, 255), thickness)

# Display the image (you can save it using cv2.imwrite)
file_path = "graph_visualization.png"
cv2.imwrite(file_path, image)

