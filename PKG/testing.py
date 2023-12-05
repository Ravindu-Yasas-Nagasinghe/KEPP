node_positions = {}
for i in range(105):
    # Assuming you want x and y coordinates to increment by 100 for each entry
    x = ((i) // 11) * 100  # 5 entries per row, increment x every 5 entries
    y = ((i) % 11) * 100   # 5 entries per row, increment y within each row
    node_positions[i] = (x, y)

print(node_positions)