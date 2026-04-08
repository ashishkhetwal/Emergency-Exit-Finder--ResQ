import heapq
import math

INT_MAX = math.inf

class PathResult:
    def __init__(self):
        self.distance = INT_MAX
        self.path = []
        self.destination_node = -1
        self.visited_count = 0

    def found(self):
        if self.distance != INT_MAX and len(self.path) > 0:
            return True
        else:
            return False

class PathFinder:
    @staticmethod
    def find_shortest_path(graph, source, destinations):
        vertices = graph.get_number_of_nodes()
        
        distance = []
        for i in range(vertices):
            distance.append(INT_MAX)
            
        parent = []
        for i in range(vertices):
            parent.append(-1)

        pq = []
        distance[source] = 0
        heapq.heappush(pq, (0, source))

        best_dest = -1
        result = PathResult()

        while pq:
            d, u = heapq.heappop(pq)
            result.visited_count += 1
            if u in destinations:
                best_dest = u
                break
            if d > distance[u]:
                continue
            for i in range(vertices):
                weight = graph.get_edge_weight(u, i)
                if weight == INT_MAX or weight == 0:
                    continue
                if distance[u] != INT_MAX and distance[u] + weight < distance[i]:
                    distance[i] = distance[u] + weight
                    parent[i] = u
                    heapq.heappush(pq, (distance[i], i))
        
        if best_dest == -1:
            return result

        result.distance = distance[best_dest]
        result.destination_node = best_dest

        current = best_dest
        while current != -1:
            result.path.append(current)
            current = parent[current]
            
        # Reverse path
        reversed_path = []
        for i in range(len(result.path) - 1, -1, -1):
            reversed_path.append(result.path[i])
        result.path = reversed_path

        return result

    @staticmethod
    def find_path_bfs(graph, source, destinations):
        from collections import deque
        result = PathResult()
        vertices = graph.get_number_of_nodes()
        
        visited = []
        for i in range(vertices):
            visited.append(False)
            
        parent = []
        for i in range(vertices):
            parent.append(-1)
        
        q = deque()
        q.append((source, 0))
        visited[source] = True
        
        best_dest = -1
        while q:
            item = q.popleft()
            u = item[0]
            hops = item[1]
            
            result.visited_count += 1
            if u in destinations:
                best_dest = u
                result.distance = hops
                break
            
            for i in range(vertices):
                weight = graph.get_edge_weight(u, i)
                if weight != 0 and weight != INT_MAX and visited[i] == False:
                    visited[i] = True
                    parent[i] = u
                    q.append((i, hops + 1))
        
        if best_dest != -1:
            result.destination_node = best_dest
            curr = best_dest
            while curr != -1:
                result.path.append(curr)
                curr = parent[curr]
                
            reversed_path = []
            for i in range(len(result.path) - 1, -1, -1):
                reversed_path.append(result.path[i])
            result.path = reversed_path
        
        return result

    @staticmethod
    def find_path_dfs(graph, source, destinations):
        result = PathResult()
        vertices = graph.get_number_of_nodes()
        
        visited = []
        for i in range(vertices):
            visited.append(False)
        
        stack = []
        initial_path = []
        initial_path.append(source)
        stack.append((source, initial_path))
        
        best_dest = -1
        while len(stack) > 0:
            item = stack.pop()
            u = item[0]
            path = item[1]
            
            if visited[u] == False:
                visited[u] = True
                result.visited_count += 1
                if u in destinations:
                    best_dest = u
                    result.distance = len(path) - 1
                    result.path = path
                    break
                
                # Push unvisited neighbors
                for i in range(vertices):
                    weight = graph.get_edge_weight(u, i)
                    if weight != 0 and weight != INT_MAX and visited[i] == False:
                        new_path = list(path)
                        new_path.append(i)
                        stack.append((i, new_path))
                        
        if best_dest != -1:
            result.destination_node = best_dest
            
        return result

    @staticmethod
    def get_all_distances(graph, source):
        vertices = graph.get_number_of_nodes()
        distance = []
        for i in range(vertices):
            distance.append(INT_MAX)
            
        pq = []
        distance[source] = 0
        heapq.heappush(pq, (0, source))
        
        while len(pq) > 0:
            item = heapq.heappop(pq)
            d = item[0]
            u = item[1]
            if d > distance[u]:
                continue
            for i in range(vertices):
                weight = graph.get_edge_weight(u, i)
                if weight == INT_MAX or weight == 0:
                    continue
                if distance[u] != INT_MAX and distance[u] + weight < distance[i]:
                    distance[i] = distance[u] + weight
                    heapq.heappush(pq, (distance[i], i))
                    
        return distance
