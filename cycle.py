matrix = [
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 1]]


def get_adgency_matrix(matrix):
    adjancy_matrix = [[0 for i in range(len(matrix) + len(matrix[0]))] for j in range(len(matrix) + len(matrix[0]))]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                adjancy_matrix[i][len(matrix) + j] = 1

    matrix = list(zip(*matrix))
    print()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                adjancy_matrix[len(matrix) + i][j] = 1
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

# Matrices d'adjacence pour des graphes avec 5 à 10 sommets

graphs = [
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ],
    [
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
]


def printMatrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


# Receive the adjacency matrix and return true if there is a cycle | false otherwise
def detect_cycle(matrix):

    # Sum of each column
    sumsColumn = [0 for i in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            sumsColumn[j] += matrix[i][j]

    # Retrieve index where the sum is 0
    memory = []
    for i in range(len(sumsColumn)):
        if sumsColumn[i] == 0:
            memory.append(i)

    # Checking if there is no more row/column to remove
    if len(memory) == 0:
        print(f"\n End Matrix \n")
        printMatrix(matrix)
        return True

    # Remove row/column of index in memory
    while len(memory) > 0:
        matrix.pop(memory[0])
        for i in range(len(matrix)):
            matrix[i].pop(memory[0])
        memory.pop(0)
        for index in range(len(memory)):
            memory[index] -= 1

    # Recursive analysis
    if len(matrix) > 0:
        return False or detect_cycle(matrix)
    else:
        print(f"\n End Matrix \n")
        printMatrix(matrix)
        return False


# cycle, vertices = find_cycle(get_adgency_matrix(matrix), start_edge)


# Affichage des matrices d'adjacence
for i, graph in enumerate(graphs, start=1):
    print(f"Graphe {i}:")
    print(f"Is there a Cycle : {detect_cycle(graph)}")
