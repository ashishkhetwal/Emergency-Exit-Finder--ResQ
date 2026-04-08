from utils import build_default_graph
from pathfinding import PathFinder

graph_data = build_default_graph()
graph = graph_data[0]
source = graph_data[1]
destinations = graph_data[2]
ow = graph_data[3]

print("Num nodes: " + str(graph.get_number_of_nodes()))
print("Source: " + str(source) + ", Destinations: " + str(destinations))

for i in range(graph.get_number_of_nodes()):
    print("Node " + str(i) + ": " + graph.get_node_name(i) + " | Floor " + str(graph.get_node_floor(i)))

result = PathFinder.find_shortest_path(graph, source, destinations)
print("Shortest path found: " + str(result.found()))
print("Hit destination: " + str(result.destination_node))
print("Distance: " + str(result.distance))
print("Path: " + str(result.path))

print("Blocking edge (1, 2)..")
# 1 is Hallway A, 2 is Main Exit, so Main Exit is blocked basically
graph.block_corridor(1, 2)
result_blocked = PathFinder.find_shortest_path(graph, source, destinations)
print("After block shortest path found: " + str(result_blocked.found()))
print("After block Hit destination: " + str(result_blocked.destination_node))
print("After block Distance: " + str(result_blocked.distance))
print("After block Path: " + str(result_blocked.path))
