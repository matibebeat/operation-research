

def get_adgency_matrix(matrix):
    adjancy_matrix = [ [0 for i in range(len(matrix)+ len(matrix[0]))] for j in range(len(matrix)+ len(matrix[0]))]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                adjancy_matrix[i][len(matrix)+j] = 1

    matrix = list(zip(*matrix))
    print()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                adjancy_matrix[len(matrix[0])+i][j] = 1
    return adjancy_matrix


def find_cycle(adj_matrix, start_edge):
    n = len(adj_matrix)
    visited = [False] * n
    path = []

    def dfs(vertex, parent):
        visited[vertex] = True
        path.append(vertex)

        for neighbor in range(n):
            if adj_matrix[vertex][neighbor]:
                if not visited[neighbor]:
                    if dfs(neighbor, vertex):
                        return True
                elif neighbor == start_edge and len(path) > 2:
                    path.append(neighbor)
                    return True

        path.pop()
        return False

    if dfs(start_edge, -1):
        cycle = []
        for i in range(len(path) - 1):
            cycle.append((path[i], path[i + 1]))
        cycle.append((path[-1], path[0]))  # Add edge back to starting vertex
        return cycle, path
    else:
        return None, None

def is_cyclic(matrix):
    adj_matrix = get_adgency_matrix(matrix)
    for i in range(len(matrix)):
        cycle, path = find_cycle(adj_matrix, i)
        if cycle:
            return True
    return False


matrix = [
    [1, 0],
    [1, 1],
    [1, 1],
    [0, 1]
]

print(get_adgency_matrix(matrix))

print(find_cycle(get_adgency_matrix(matrix),2)) 
"""
[[0, 0, 0, 0, 1, 0], 
[0, 0, 0, 0, 1, t], 
[0, 0, 0, 0, f, 1], 
[0, 0, 0, 0, 0, 1], 
[1, s, 1, 0, 0, 0], 
[0, 1, q, 1, 0, 0]]




([(2, 4), (4, 1), (1, 5), (5, 2), (2, 2)], [2, 4, 1, 5, 2])

(2,0)->(0,1)->(1,1)->(1)->(2,0)
                    
-4      -4     -4      

"""


