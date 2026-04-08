import os
from graph import Graph

def parse_input_file(content):
    lines = []
    for l in content.strip().splitlines():
        if l.strip() != "":
            lines.append(l.strip())
            
    idx = 0

    num_nodes = int(lines[idx])
    idx += 1
    graph = Graph(num_nodes)

    for i in range(num_nodes):
        parts = lines[idx].split("|")
        name = parts[0].strip()
        
        if len(parts) > 1:
            floor = int(parts[1].strip())
        else:
            floor = 0
            
        graph.set_node_name(name, i)
        graph.set_node_floor(floor, i)
        idx += 1

    num_edges = int(lines[idx])
    idx += 1
    original_weights = {}

    for i in range(num_edges):
        parts = lines[idx].split()
        idx += 1
        u = int(parts[0])
        v = int(parts[1])
        w = int(parts[2])
        graph.add_edge(u, v, w)
        
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)
        original_weights[edge] = w

    source = int(lines[idx])
    idx += 1
    
    destinations = []
    for x in lines[idx].split():
        destinations.append(int(x))

    return graph, source, destinations, original_weights

def build_default_graph():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_input_file(content)
