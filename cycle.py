
matrix = [
 [1, 0, 0],
 [1, 0, 1],
 [1, 1, 1]]
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
                adjancy_matrix[len(matrix)+i][j] = 1
    return adjancy_matrix



def find_path(adj_matrix, start_edge):
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

# Example adjacency matrix

start_edge = 6  # Starting edge index


print(get_adgency_matrix(matrix))

#cycle, vertices = find_cycle(get_adgency_matrix(matrix), start_edge)

if cycle:
    print("Cycle found:", cycle)
    print("Vertices used:", vertices)
else:
    print("No cycle found starting from the given edge.")



