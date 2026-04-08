from sorting import merge_sort

class DisjointSet:
    def __init__(self, n):
        self.parent = []
        for i in range(n):
            self.parent.append(i)
            
        self.rank = []
        for i in range(n):
            self.rank.append(0)
        
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        else:
            return False

def kruskal_mst(graph):
    edges = graph.get_all_edges()
    
    def get_weight(item):
        return item[2]
        
    edges = merge_sort(edges, key=get_weight)
    
    ds = DisjointSet(graph.get_number_of_nodes())
    mst = []
    total_cost = 0
    
    for edge in edges:
        u = edge[0]
        v = edge[1]
        w = edge[2]
        
        if graph.is_blocked(u, v):
            continue
        if ds.union(u, v):
            mst.append((u, v, w))
            total_cost += w
            
    return mst, total_cost
