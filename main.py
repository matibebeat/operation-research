
import sys
import time
import random
# Our libraries
from maths import resolve_equation, solve_2nd_order_system
from graph import find_cycle, get_adgency_matrix , is_cyclic



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

def delayed_print(text):
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
        
    def init_random(self, x, y):
        self.orders = [random.randint(1, 100) for _ in range(x)]
        self.provisions = [random.randint(1, 100) for _ in range(y)]
        self.matrix = [[random.randint(1, 100) for _ in range(x)] for j in range(y)]
        self.costs = None

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


        while True:
            #check if the number of edges is equal to the number of vertices -1
            vertices = 0
            edges = 0
            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    if solution[i][j] != 0:
                        edges += 1
            
            for i in range(len(solution)):
                if sum(solution[i]) != 0:
                    vertices += 1
            for i in range(len(solution[0])):
                if sum([solution[j][i] for j in range(len(solution))]) != 0:
                    vertices += 1

            solution_graph = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    if solution[i][j] != 0:
                        solution_graph[i][j] = 1

            
            debug= 0
            while edges != vertices - 1:
                #!we need to add an edge to the solution
                #we will find the edge with the lowest transport cost, check if we add it we will have a cycle
                min_cost = float('inf')
                min_indice = []
                for i in range(len(self.matrix)):
                    for j in range(len(self.matrix[i])):
                        if self.matrix[i][j] < min_cost and solution[i][j] == 0:
                            #we make temp_solution wich is the solution with the edge added with a value of 1 for each edge
                            temp_solution = [[0 for i in range(len(solution[0]))] for j in range(len(solution))]
                            for k in range(len(temp_solution)):
                                for l in range(len(temp_solution[k])):
                                    if temp_solution[k][l] != 0:
                                        temp_solution[k][l] = 1
                            temp_solution[i][j] = 1
                            if not is_cyclic(temp_solution):
                                min_cost = self.matrix[i][j]
                                min_indice = [i, j]

                debug = 0   
                solution_graph[min_indice[0]][min_indice[1]] = 1

                vertices = 0
                edges = 0
                for i in range(len(solution_graph)):
                    for j in range(len(solution_graph[i])):
                        if solution_graph[i][j] != 0:
                            edges += 1
            
                for i in range(len(solution_graph)):
                    if sum(solution_graph[i]) != 0:
                        vertices += 1
                for i in range(len(solution_graph[0])):
                    if sum([solution_graph[j][i] for j in range(len(solution_graph))]) != 0:
                        vertices += 1

                
            #first we need to check if the solution is degenerate
            #if is_degenerate(solution):
                #raise Exception("The solution is degenerate")
            string = ""
            system = []
            for i in range(len(solution_graph)):
                for j in range(len(solution_graph[i])):
                    if solution_graph[i][j] != 0:
                        string = f"y{i}-x{j}={self.matrix[i][j]}"
                        system.append(string)

            soluce = solve_2nd_order_system(system, {"y0": 0})
            potentials_row = [None for i in range(len(self.provisions))]
            potentials_col = [None for i in range(len(self.orders))]
            potentials_row[0] = 0

            for elem in soluce:
                if "x" in elem:
                    potentials_col[int(elem[1])] = soluce[elem]
                else:
                    potentials_row[int(elem[1])] = soluce[elem]

            print(potentials_row)
            print(potentials_col)

            #copute the marginal costs
            potential_costs = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
            for i in range(len(self.provisions)):
                for j in range(len(self.orders)):
                    potential_costs[i][j] = potentials_row[i] - potentials_col[j]

            marginal_costs = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
            for i in range(len(self.provisions)):
                for j in range(len(self.orders)):
                    marginal_costs[i][j] = self.matrix[i][j] - potential_costs[i][j]

            #we find the edge with the lowest marginal cost
            min_cost = float('inf')
            min_indice = []
            for i in range(len(self.provisions)):
                for j in range(len(self.orders)):
                    if marginal_costs[i][j] < min_cost:
                        min_cost = marginal_costs[i][j]
                        min_indice = [i, j]

            if min_cost >= 0:
                break

            #we hade the edge to the solution
            solution[min_indice[0]][min_indice[1]] = 1
            solution_graph[min_indice[0]][min_indice[1]] = 1


            start_edge = min_indice[1] 
            cycle, vertices = find_cycle(get_adgency_matrix(solution_graph), start_edge)

            if not cycle:
                start_edge = min_indice[0]+ len(self.orders)
                cycle, vertices = find_cycle(get_adgency_matrix(solution_graph), start_edge)

            #find (i,j) in the cycle
            path=[]
            for i in range(len(cycle)):
                #find the max value of cycle[i]
                if cycle[i][0] < cycle[i][1] :
                    y= cycle[i][1]-len(self.provisions)
                    x= cycle[i][0]
                else:
                    x= cycle[i][0]-len(self.provisions)
                    y= cycle[i][1]
                path.append((y,x))
            
            #if we have a negative value in the cycle we need to add 




            orders = self.orders.copy()
            provisions = self.provisions.copy()

            

            #!on en est là
            #! jusqu'ici tout va bien


            max_value_to_add = float('inf')
            for i in range(len(path[1:])):
                if i % 2 == 0:
                    #we need to find the min value to add by taking care of the orders and provisions and all the other values in the path
                    max_value_to_add = min(max_value_to_add, min(provisions[path[i][0]], orders[path[i][1]]))




                else:
                    max_value_to_add = min(max_value_to_add, solution[path[i][1]][path[i][0]])
                
            for i in range(len(path[1:])):
                if i % 2 == 0:
                    solution[path[i][0]][path[i][1]] += max_value_to_add
                else:
                    solution[path[i][1]][path[i][0]] -= max_value_to_add
            
            solution[min_indice[0]][min_indice[1]] -=1
                


                

            
                

        

            pass
        return solution

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


#test = transportation_problem('files/problem_9.txt')
#print(test.north_west_corner())
#print(test.stepping_stone())

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


"""
for i in range(3,13):
    print("done" + str(i))
    with open(f'files/problem_{i}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
    test = transportation_problem(f'files/problem_{i}.txt')
    matrix = test.matrix
    orders = test.orders
    provisions = test.provisions
    North_west = test.stepping_stone()
    pass"""
    
test = transportation_problem('files/problem_1.txt')
print(test.north_west_corner())
print(test.stepping_stone())


    


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


