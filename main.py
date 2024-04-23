
import sys
import time
import random

def is_E_V(matrix):
    Number_vertices = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                Number_vertices += 1
    Number_edges = len(matrix) + len(matrix[0]) 
    if Number_vertices == Number_edges - 1:
        return False
    return True

def is_acyclic(proposal):
    num_vertices = len(proposal)
    visited = set()

    def dfs(node, visited, proposal):
        if node in visited:
            return False
        visited.add(node)
        for i in range(num_vertices):
            if proposal[node][i] == 1:
                if not dfs(i, visited, proposal):
                    return False
        visited.remove(node)
        return True
    
    for i in range(num_vertices):
        if not dfs(i, visited, proposal):
            return False
    return True

def resolve_equation(a,b,c):
    if a == none and b == none:
        return a,b,c
    if a == none:
        return c+b, b, c
    if b == none:
        return a, a-c, c
    


def delayed_print(text, delay=0.05):
    text = str(text)
    for char in text:
        #delay rand between 0.1 and 0.01
        delay = random.uniform(0.1, 0.01)
        print(char, end='', flush=True)
        time.sleep(delay)
        
def is_degenerate(matrix):
    return (is_E_V(matrix) and is_acyclic(matrix))




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

    #constructor of the class with a str file as parameter
    def __init__(self, file=None, x=5, y=5):
        if file == None:
            self.init_random(x,y)
            return
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
            self.matrix = []
            for elem in content[:-1]:
                self.matrix.append(list(map(int, elem.split()[:-1])))
        self.costs = None
        #self.north_west_solution = self.north_west_corner()
        #delayed_print(self.compute_cost(self.north_west_solution))
        
        #delayed_print(self.ballas_hammer_charles())
        
    def init_random(self, x, y):
        self.orders = [random.randint(1, 100) for i in range(x)]
        self.provisions = [random.randint(1, 100) for i in range(y)]
        self.matrix = [[random.randint(1, 100) for i in range(x)] for j in range(y)]
        self.costs = None
        #self.north_west_solution = self.north_west_corner()
        #delayed_print(self.compute_cost(self.north_west_solution))
        
        #delayed_print(self.ballas_hammer_charles())

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
                
        delayed_print("cc"+str(solution))
        
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
        solution = self.north_west_corner()

        #first we need to check if the solution is degenerate
        if is_degenerate(solution):
            throw("The solution is degenerate")
        #create a matrix of size of solution with [] where in solution there is a value > 0 and none otherwise
        print(solution)
        potential_cost = [[None if solution[i][j] == 0 else [] for j in range(len(self.orders))] for i in range(len(self.provisions))]
        print(potential_cost)

        #calculate the potentials
        potentials_row = [None for i in range(len(self.provisions))]
        potentials_col = [None for i in range(len(self.orders))]
        potentials_row[0] = 0
        #calculate the potentials
        for i in range(len(potentials_row)):
            for j in range(len(potentials_col)):
                if solution[i][j] != 0:
                    if potentials_row[i] != None:
                        potentials_col[j] =  potentials_row[i] -self.matrix[i][j]
                    else:
                        potentials_row[i] =  potentials_col[j] -self.matrix[i][j]
        
        print(potentials_row)
        print(potentials_col)

        #for each cell in the solution, calculate the potential cost
        for i in range(len(solution)):
            for j in range(len(solution[i])):
                
                potential_cost[i][j] =  potentials_row[i] - potentials_col[j]
        
        print(potential_cost)
                

        

        pass

    def __str__(self): #TODO : implement the __str__ method -> @Mathieu fait nous des beaux tableaux 
        """
        cost cost cost cost cost prov
        cost cost cost cost cost prov
        cost cost cost cost cost prov
        cost cost cost cost cost prov
        ord ord ord ord ord
        """
        string = ""
        for i in range(len(self.matrix)):
            string += " ".join(str(x) for x in self.matrix[i]) + " " + str(self.provisions[i]) + "\n"
        string += " ".join(str(x) for x in self.orders)
        return string
        



    

        


        return f"Orders : {self.orders}\nProvisions : {self.provisions}\nMatrix : {self.matrix}"


test = transportation_problem('files/tp_1.txt')
test.stepping_stone()

"""
test = transportation_problem('files/tp_1.txt')

global problems
problems =[]

problems.append(test)

the = transportation_problem(x = 5, y= 5)
print(the)

"""
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







    


# Test the function
proposal_with_cycle = [
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 0]
]

prop2 =[
    [1,1,0],
    [1,1,0],
    [0,1,1]
]


def is_connected():#! c'est drole
    fetch("google.com")
    if response == 200:
        return True








"""
matrix =[ [65,10,4,0], [0,30,10,0], [0,0,50,20] ]

print(is_degenerate(matrix))

print(is_acyclic(proposal_with_cycle))
print(is_acyclic(prop2))


for i in range(1,13):
    print("done" + str(i))
    with open(f'files/problem_{i}.txt', 'r') as f:
        lines = f.readlines()
        print (lines)
    #open each file and write it in the file
    with open(f'files/problem_{i}.txt', 'w+') as f:
        f.write(lines[0])
        #loop over the lines exept the first one and write them in the file
        for line in lines[1:]:
            line = line.split()
            f.write(" ".join(line[1:]) + "\n")
        
        """
    
"""
    print(lines)
        f.write(lines[0])
        #loop over the lines exept the first one and write them in the file
        for line in lines[1:]:
            line = line.split()
            f.write(" ".join(line[:-1]) + "\n")
    """