def calculate_penalty(costs):
    penalties_row = []
    penalties_col = []
    # Calculate penalties for rows
    for row in costs:
        sorted_row = sorted(row)
        penalty = sorted_row[1] - sorted_row[0]
        penalties_row.append(penalty)

    # Calculate penalties for columns
    transposed_costs = list(zip(*costs))
    for col in transposed_costs:
        sorted_col = sorted(col)
        penalty = sorted_col[1] - sorted_col[0]
        penalties_col.append(penalty)

    return penalties_row,penalties_col # Return the penalties for rows and columns

def select_max_penalty(costs):
    penalties_row, penalties_col = calculate_penalty(costs)
    
    max_penalty = max(max(penalties_row), max(penalties_col))
    max_indices = []

    for i, penalty in enumerate(penalties_row):
        if penalty == max_penalty:
            max_indices.append(('row', i))
    
    for i, penalty in enumerate(penalties_col):
        if penalty == max_penalty:
            max_indices.append(('col', i))

    max_capacity = float('-inf')
    selected_index = None
    for index in max_indices:
        if index[0] == 'row':
            min_capacity = min(costs[index[1]])
        else:
            min_capacity = min(row[index[1]] for row in costs)
        
        if min_capacity > max_capacity:
            selected_index = index
            max_capacity = min_capacity
    
    return selected_index


class transportation_problem():
    def __init__(self, file):
        """
        the content of files are like that: 
        30 20 20 450
        10 50 20 250
        50 40 30 250
        30 20 30 450
        500 600 300

        last column is the provision values
        last line is the order values
        the other values are the costs of transportation
        """

        with open(file, 'r') as f:
            content = f.read().split('\n')
            self.orders = list(map(int, content[-1].split()))
            self.provisions=[]
            for elem in content[:-1]:
                self.provisions.append(int(elem.split()[-1]))
            print("prov"+str(self.provisions))
            print(self.orders)
            self.matrix = []
            for elem in content[:-1]:
                self.matrix.append(list(map(int, elem.split()[:-1])))
            print(self.matrix)
        self.costs = None
        #self.north_west_solution = self.north_west_corner()
        #print(self.compute_cost(self.north_west_solution))
        print(self.ballas_hammer_charles())
        



    def north_west_corner(self):
        #a 2D list that contains the solution of the problem
        orders = self.orders.copy()
        provisions = self.provisions.copy()

        solution = [[None for i in range(len(orders))] for j in range(len(self.provisions))]
        #while there are still orders to be satisfied
        for i in range(len(provisions)):
            for j in range(len(orders)):
                solution[i][j] = min(provisions[i], orders[j])
                provisions[i] -= solution[i][j]
                orders[j] -= solution[i][j]
                
        print("cc"+str(solution))
        
        return solution


    def ballas_hammer_charles(self): #!important : replace all variables nammes 
        orders = self.orders.copy()
        provisions = self.provisions.copy()
        costs = self.matrix.copy()
        allocated_costs = [[0] * len(row) for row in costs]

        while True:
            selected_index = select_max_penalty(costs)
            if selected_index is None:
                break
        
            if selected_index[0] == 'row':
                selected_row = selected_index[1]
                min_cost = min(costs[selected_row])
                min_cost_index = costs[selected_row].index(min_cost)
                max_supply = min(provisions[selected_row], orders[min_cost_index])
                allocated_costs[selected_row][min_cost_index] += max_supply  # Allocate to the separate table
                provisions[selected_row] -= max_supply
                orders[min_cost_index] -= max_supply
                costs[selected_row][min_cost_index] = float('inf')  # Mark the cell as used by setting it to infinity
            else:
                selected_col = selected_index[1]
                col_values = [row[selected_col] for row in costs]
                min_cost = min(col_values)
                min_cost_index = col_values.index(min_cost)
                max_demand = min(provisions[min_cost_index], orders[selected_col])
                allocated_costs[min_cost_index][selected_col] += max_demand  # Allocate to the separate table
                provisions[min_cost_index] -= max_demand
                orders[selected_col] -= max_demand
                costs[min_cost_index][selected_col] = float('inf')  # Mark the cell as used by setting it to infinity

        return allocated_costs
        

    def compute_cost(self, solution):
        cost = 0
        for i in range(len(self.provisions)):
            for j in range(len(self.orders)):
                cost += solution[i][j] * self.matrix[i][j]
        return cost

    def stepping_stone(self):
        #TODO : implement the stepping stone method
        """
        Solving algorithm : the stepping-stone method with potential.
            ⋆ Test whether the proposition is acyclic : we’ll use a Breadth-first algorithm. During the algorithm run, as the vertices are discovered, we check that we’re returning to a previously
               visited vertex and that this vertex isn’t the parent of the current vertex ; if it is, then a cycle exists. The cycle is then displayed.
            ⋆ Transportation maximization if a cycle has been detected. The conditions for each box are
               displayed. Then we display the deleted edge (possibly several) at the end of maximization.
            ⋆ Test whether the proposition is connected : we’ll use a Breadth-first algorithm. If it is not
               connected : display of all connected sub-graphs.
            ⋆ Modification of the graph if it is unconnected, until a non-degenerate proposition is obtained.
            ⋆ Calculation and display of potentials per vertex.
            ⋆ Display of both potential costs table and marginal costs table. Possible detection of the best
               improving edge.
            ⋆ Add this improving edge to the transport proposal, if it has been detected.
        """

        pass

    def __str__(self): #TODO : implement the __str__ method -> @Mathieu fait nous des beaux tableaux 
        return f"Orders : {self.orders}\nProvisions : {self.provisions}\nMatrix : {self.matrix}"


test = transportation_problem('files/tp_1.txt')

global problems
problems =[]

problems.append(test)



def menu():
    """
    Start
        While the user decides to test a transportation problem, do :
            Choice of the problem number to be processed.
            Read the table of constraints from a file and store it in memory
            Create the corresponding matrice representing this table and display it
            Ask the user to choose the algorithm to fix the initial proposal and execute it.
            Display the elements mentioned above when running the two algorithms.
            Run the stepping-stone method with potential, displaying at each iteration :
                ⋆ Displays the transport proposal and the total transport cost.
                ⋆ Test to know if the transport proposal is degenerate.
                ⋆ Modification of the transport graph to obtain a tree, in the cyclic or non connected.
                ⋆ Potentials calculation and display.
                ⋆ Table display : potential costs and marginal costs.
                    ⋆ If not optimal :
                        Displays the edge to be added.
                        Transport maximization on the formed cycle and a new iteration.
                    ⋆ Else exit the loop
                        ⋆ End if
            Display the minimal transportation proposal and its cost.
            Suggest to the user that he/she should change transportation problem
        End while
    End

    """


menu()


