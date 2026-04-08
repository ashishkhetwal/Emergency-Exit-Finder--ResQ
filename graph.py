import math

INT_MAX = math.inf

class Graph:
    def __init__(self, num_nodes):
        self.number_of_nodes = num_nodes
        
        self.adjacency_matrix = []
        for i in range(num_nodes):
            row = []
            for j in range(num_nodes):
                row.append(0)
            self.adjacency_matrix.append(row)
            
        self.node_names = []
        for i in range(num_nodes):
            name = "Node " + str(i)
            self.node_names.append(name)
            
        self.node_floors = []
        for i in range(num_nodes):
            self.node_floors.append(0)
            
        self.blocked_edges = set()

    def set_node_floor(self, floor, node):
        self.node_floors[node] = floor

    def get_node_floor(self, node):
        return self.node_floors[node]

    def add_edge(self, u, v, wt):
        self.adjacency_matrix[u][v] = wt
        self.adjacency_matrix[v][u] = wt

    def block_corridor(self, u, v):
        self.adjacency_matrix[u][v] = INT_MAX
        self.adjacency_matrix[v][u] = INT_MAX
        
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)
        self.blocked_edges.add(edge)

    def unblock_corridor(self, u, v, original_weight):
        self.adjacency_matrix[u][v] = original_weight
        self.adjacency_matrix[v][u] = original_weight
        
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)
        self.blocked_edges.discard(edge)

    def set_node_name(self, name, node):
        self.node_names[node] = name

    def get_node_name(self, node):
        return self.node_names[node]

    def get_edge_weight(self, u, v):
        return self.adjacency_matrix[u][v]

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def is_blocked(self, u, v):
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)
            
        if edge in self.blocked_edges:
            return True
        else:
            return False

    def get_all_edges(self):
        edges = []
        for i in range(self.number_of_nodes):
            for j in range(i + 1, self.number_of_nodes):
                w = self.adjacency_matrix[i][j]
                if w != 0 and w != INT_MAX:
                    edges.append((i, j, w))
        return edges
