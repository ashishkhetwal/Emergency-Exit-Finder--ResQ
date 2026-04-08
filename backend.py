# import heapq
# import math
# import os

# INT_MAX = math.inf


# class Graph:
#     def __init__(self, num_nodes):
#         self.number_of_nodes = num_nodes
        
#         self.adjacency_matrix = []
#         for i in range(num_nodes):
#             row = []
#             for j in range(num_nodes):
#                 row.append(0)
#             self.adjacency_matrix.append(row)
            
#         self.node_names = []
#         for i in range(num_nodes):
#             name = "Node " + str(i)
#             self.node_names.append(name)
            
#         self.node_floors = []
#         for i in range(num_nodes):
#             self.node_floors.append(0)
            
#         self.blocked_edges = set()

#     def set_node_floor(self, floor, node):
#         self.node_floors[node] = floor

#     def get_node_floor(self, node):
#         return self.node_floors[node]

#     def add_edge(self, u, v, wt):
#         self.adjacency_matrix[u][v] = wt
#         self.adjacency_matrix[v][u] = wt

#     def block_corridor(self, u, v):
#         self.adjacency_matrix[u][v] = INT_MAX
#         self.adjacency_matrix[v][u] = INT_MAX
        
#         if u < v:
#             edge = (u, v)
#         else:
#             edge = (v, u)
#         self.blocked_edges.add(edge)

#     def unblock_corridor(self, u, v, original_weight):
#         self.adjacency_matrix[u][v] = original_weight
#         self.adjacency_matrix[v][u] = original_weight
        
#         if u < v:
#             edge = (u, v)
#         else:
#             edge = (v, u)
#         self.blocked_edges.discard(edge)

#     def set_node_name(self, name, node):
#         self.node_names[node] = name

#     def get_node_name(self, node):
#         return self.node_names[node]

#     def get_edge_weight(self, u, v):
#         return self.adjacency_matrix[u][v]

#     def get_number_of_nodes(self):
#         return self.number_of_nodes

#     def is_blocked(self, u, v):
#         if u < v:
#             edge = (u, v)
#         else:
#             edge = (v, u)
            
#         if edge in self.blocked_edges:
#             return True
#         else:
#             return False

#     def get_all_edges(self):
#         edges = []
#         for i in range(self.number_of_nodes):
#             for j in range(i + 1, self.number_of_nodes):
#                 w = self.adjacency_matrix[i][j]
#                 if w != 0 and w != INT_MAX:
#                     edges.append((i, j, w))
#         return edges


# class PathResult:
#     def __init__(self):
#         self.distance = INT_MAX
#         self.path = []
#         self.destination_node = -1
#         self.visited_count = 0

#     def found(self):
#         if self.distance != INT_MAX and len(self.path) > 0:
#             return True
#         else:
#             return False

# class PathFinder:
#     @staticmethod
#     def find_shortest_path(graph, source, destinations):
#         vertices = graph.get_number_of_nodes()
        
#         distance = []
#         for i in range(vertices):
#             distance.append(INT_MAX)
            
#         parent = []
#         for i in range(vertices):
#             parent.append(-1)

#         pq = []
#         distance[source] = 0
#         heapq.heappush(pq, (0, source))

#         best_dest = -1
#         result = PathResult()

#         while pq:
#             d, u = heapq.heappop(pq)
#             result.visited_count += 1
#             if u in destinations:
#                 best_dest = u
#                 break
#             if d > distance[u]:
#                 continue
#             for i in range(vertices):
#                 weight = graph.get_edge_weight(u, i)
#                 if weight == INT_MAX or weight == 0:
#                     continue
#                 if distance[u] != INT_MAX and distance[u] + weight < distance[i]:
#                     distance[i] = distance[u] + weight
#                     parent[i] = u
#                     heapq.heappush(pq, (distance[i], i))
        
#         if best_dest == -1:
#             return result

#         result.distance = distance[best_dest]
#         result.destination_node = best_dest

#         current = best_dest
#         while current != -1:
#             result.path.append(current)
#             current = parent[current]
            
#         # Reverse path
#         reversed_path = []
#         for i in range(len(result.path) - 1, -1, -1):
#             reversed_path.append(result.path[i])
#         result.path = reversed_path

#         return result

#     @staticmethod
#     def find_path_bfs(graph, source, destinations):
#         from collections import deque
#         result = PathResult()
#         vertices = graph.get_number_of_nodes()
        
#         visited = []
#         for i in range(vertices):
#             visited.append(False)
            
#         parent = []
#         for i in range(vertices):
#             parent.append(-1)
        
#         q = deque()
#         q.append((source, 0))
#         visited[source] = True
        
#         best_dest = -1
#         while q:
#             item = q.popleft()
#             u = item[0]
#             hops = item[1]
            
#             result.visited_count += 1
#             if u in destinations:
#                 best_dest = u
#                 result.distance = hops
#                 break
            
#             for i in range(vertices):
#                 weight = graph.get_edge_weight(u, i)
#                 if weight != 0 and weight != INT_MAX and visited[i] == False:
#                     visited[i] = True
#                     parent[i] = u
#                     q.append((i, hops + 1))
        
#         if best_dest != -1:
#             result.destination_node = best_dest
#             curr = best_dest
#             while curr != -1:
#                 result.path.append(curr)
#                 curr = parent[curr]
                
#             reversed_path = []
#             for i in range(len(result.path) - 1, -1, -1):
#                 reversed_path.append(result.path[i])
#             result.path = reversed_path
        
#         return result

#     @staticmethod
#     def find_path_dfs(graph, source, destinations):
#         result = PathResult()
#         vertices = graph.get_number_of_nodes()
        
#         visited = []
#         for i in range(vertices):
#             visited.append(False)
        
#         stack = []
#         initial_path = []
#         initial_path.append(source)
#         stack.append((source, initial_path))
        
#         best_dest = -1
#         while len(stack) > 0:
#             item = stack.pop()
#             u = item[0]
#             path = item[1]
            
#             if visited[u] == False:
#                 visited[u] = True
#                 result.visited_count += 1
#                 if u in destinations:
#                     best_dest = u
#                     result.distance = len(path) - 1
#                     result.path = path
#                     break
                
#                 # Push unvisited neighbors
#                 for i in range(vertices):
#                     weight = graph.get_edge_weight(u, i)
#                     if weight != 0 and weight != INT_MAX and visited[i] == False:
#                         new_path = list(path)
#                         new_path.append(i)
#                         stack.append((i, new_path))
                        
#         if best_dest != -1:
#             result.destination_node = best_dest
            
#         return result

#     @staticmethod
#     def get_all_distances(graph, source):
#         vertices = graph.get_number_of_nodes()
#         distance = []
#         for i in range(vertices):
#             distance.append(INT_MAX)
            
#         pq = []
#         distance[source] = 0
#         heapq.heappush(pq, (0, source))
        
#         while len(pq) > 0:
#             item = heapq.heappop(pq)
#             d = item[0]
#             u = item[1]
#             if d > distance[u]:
#                 continue
#             for i in range(vertices):
#                 weight = graph.get_edge_weight(u, i)
#                 if weight == INT_MAX or weight == 0:
#                     continue
#                 if distance[u] != INT_MAX and distance[u] + weight < distance[i]:
#                     distance[i] = distance[u] + weight
#                     heapq.heappush(pq, (distance[i], i))
                    
#         return distance


# def parse_input_file(content):
#     lines = []
#     for l in content.strip().splitlines():
#         if l.strip() != "":
#             lines.append(l.strip())
            
#     idx = 0

#     num_nodes = int(lines[idx])
#     idx += 1
#     graph = Graph(num_nodes)

#     for i in range(num_nodes):
#         parts = lines[idx].split("|")
#         name = parts[0].strip()
        
#         if len(parts) > 1:
#             floor = int(parts[1].strip())
#         else:
#             floor = 0
            
#         graph.set_node_name(name, i)
#         graph.set_node_floor(floor, i)
#         idx += 1

#     num_edges = int(lines[idx])
#     idx += 1
#     original_weights = {}

#     for i in range(num_edges):
#         parts = lines[idx].split()
#         idx += 1
#         u = int(parts[0])
#         v = int(parts[1])
#         w = int(parts[2])
#         graph.add_edge(u, v, w)
        
#         if u < v:
#             edge = (u, v)
#         else:
#             edge = (v, u)
#         original_weights[edge] = w

#     source = int(lines[idx])
#     idx += 1
    
#     destinations = []
#     for x in lines[idx].split():
#         destinations.append(int(x))

#     return graph, source, destinations, original_weights


# def build_default_graph():
#     file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input.txt')
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
#     return parse_input_file(content)

# # --- Additional Algorithms ---

# def _default_key(x):
#     return x

# def merge_sort(arr, key=None):
#     if key is None:
#         key = _default_key
        
#     if len(arr) <= 1:
#         return arr
    
#     mid = len(arr) // 2
    
#     left_half = []
#     for i in range(mid):
#         left_half.append(arr[i])
        
#     right_half = []
#     for i in range(mid, len(arr)):
#         right_half.append(arr[i])
        
#     left = merge_sort(left_half, key)
#     right = merge_sort(right_half, key)
    
#     result = []
#     i = 0
#     j = 0
#     while i < len(left) and j < len(right):
#         if key(left[i]) <= key(right[j]):
#             result.append(left[i])
#             i += 1
#         else:
#             result.append(right[j])
#             j += 1
            
#     while i < len(left):
#         result.append(left[i])
#         i += 1
        
#     while j < len(right):
#         result.append(right[j])
#         j += 1
        
#     return result

# class DisjointSet:
#     def __init__(self, n):
#         self.parent = []
#         for i in range(n):
#             self.parent.append(i)
            
#         self.rank = []
#         for i in range(n):
#             self.rank.append(0)
        
#     def find(self, i):
#         if self.parent[i] == i:
#             return i
#         self.parent[i] = self.find(self.parent[i])
#         return self.parent[i]
        
#     def union(self, i, j):
#         root_i = self.find(i)
#         root_j = self.find(j)
#         if root_i != root_j:
#             if self.rank[root_i] < self.rank[root_j]:
#                 self.parent[root_i] = root_j
#             elif self.rank[root_i] > self.rank[root_j]:
#                 self.parent[root_j] = root_i
#             else:
#                 self.parent[root_j] = root_i
#                 self.rank[root_i] += 1
#             return True
#         else:
#             return False

# def kruskal_mst(graph):
#     edges = graph.get_all_edges()
    
#     def get_weight(item):
#         return item[2]
        
#     edges = merge_sort(edges, key=get_weight)
    
#     ds = DisjointSet(graph.get_number_of_nodes())
#     mst = []
#     total_cost = 0
    
#     for edge in edges:
#         u = edge[0]
#         v = edge[1]
#         w = edge[2]
        
#         if graph.is_blocked(u, v):
#             continue
#         if ds.union(u, v):
#             mst.append((u, v, w))
#             total_cost += w
            
#     return mst, total_cost
