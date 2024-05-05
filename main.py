
import sys
import time
import random
from prettytable import PrettyTable,DOUBLE_BORDER
# Our libraries
from maths import resolve_equation, solve_2nd_order_system
from graph import find_cycle, get_adgency_matrix , is_cyclic, adjency
from cycle import detect_cycle
from math import isnan
from steppingstone import *
import numpy as np
import matplotlib.pyplot as plt





def find_minimum_in_matrix(matrix):
    minimum = float('inf')
    minimum_index = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < minimum:
                minimum = matrix[i][j]
                minimum_index = [i, j]
    return minimum, minimum_index



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
    for i in matrix: print
    #print("1 :",is_E_V(matrix))
    #print("2 :",detect_cycle(adjency(matrix)))
    return is_E_V(matrix) or detect_cycle((adjency(matrix)))


def fix_cycle(self,matrix):
    cycle, path = find_cycle(adjency(matrix),0)
    #print("cycle",cycle)
    #print("path",path)



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
    penalty_row_sans_nan_pcq_il_est_alergique = []
    penalty_col_sans_nan_pcq_il_est_alergique = []
    for i in range(len(penalties_row)):
        if penalties_row[i] == penalties_row[i] :
            penalty_row_sans_nan_pcq_il_est_alergique.append(penalties_row[i])
    for i in range(len(penalties_col)):
        if penalties_col[i] == penalties_col[i] :
            penalty_col_sans_nan_pcq_il_est_alergique.append(penalties_col[i])

    if len(penalty_row_sans_nan_pcq_il_est_alergique) == 1:
        max_penalty_row = penalty_row_sans_nan_pcq_il_est_alergique[0]
    else:
        max_penalty_row = max(penalty_row_sans_nan_pcq_il_est_alergique)
    if len(penalty_col_sans_nan_pcq_il_est_alergique) == 1:
        max_penalty_col = penalty_col_sans_nan_pcq_il_est_alergique[0]
    else:
        max_penalty_col = max(penalty_col_sans_nan_pcq_il_est_alergique)
    max_penalty = max(max(penalty_row_sans_nan_pcq_il_est_alergique), max(penalty_col_sans_nan_pcq_il_est_alergique))

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

        if min_capacity < max_capacity or max_capacity == float('-inf'):
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

        self.cost_matrice = copy.deepcopy(self.matrix)
        self.costs = None


    def fix_E_V(self, matrix):
        #for i in self.matrix: print (i)
        matrix_cost = copy.deepcopy(self.matrix)
        stop = 1
        while(stop):
            minimum, minimum_index = find_minimum_in_matrix(matrix_cost)
            # if meme index ballas = 0 non relié
            if matrix[minimum_index[0]][minimum_index[1]] == 0:
                matrix[minimum_index[0]][minimum_index[1]] = 0.00000000000000001
                if not detect_cycle(adjency(matrix)):
                    stop = 0
                else:    
                    matrix[minimum_index[0]][minimum_index[1]] = 0
            matrix_cost[minimum_index[0]][minimum_index[1]] = 999    
        return matrix
    
    def init_random(self, livreur, client):
        self.orders = []
        self.provisions = []
        total_demand = 0
        total_stock = 0

        for _ in range(livreur):
            stock = random.randint(1, 100)
            self.provisions.append(stock)
            total_stock += stock

        for _ in range(client):
            demand = random.randint(1, total_stock) // client
            self.orders.append(demand)
            total_demand += demand

        # Correction pour s'assurer que la somme des demandes soit égale à la somme des stocks
        diff = total_stock - total_demand
        if diff > 0:
            self.orders[0] += diff
        elif diff < 0:
            self.orders[0] -= diff

        self.matrix = [[random.randint(1, 100) for _ in range(client)] for _ in range(livreur)]
        self.costs = None

        return self

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

        table = PrettyTable()

        num_provisions = len(solution)
        num_orders = len(solution[0])

        # Ajout de la colonne pour les noms des provisions
        table.add_column("", ["S " + str(i + 1) for i in range(num_provisions)])

        # Ajout des colonnes pour chaque commande
        for i in range(num_orders):
            table.add_column("L " + str(i + 1), [solution[j][i] for j in range(num_provisions)])
        table.add_column("Provisions", self.provisions)
        total_orders = sum(self.orders)
        table.add_row(["Orders"] + self.orders + [total_orders])
        # Affichage du tableau
        table.set_style(DOUBLE_BORDER)
        print(table)   
               
                
        
        
        return solution

    def ballas_hammer(self): 
        orders = self.orders.copy()
        provisions = self.provisions.copy()
        costs = copy.deepcopy(self.matrix)
        allocated_costs = [[0] * len(row) for row in costs]
        
        while not all(all(cell == float('inf') for cell in row) for row in costs):
            try:
                selected_index = select_max_penalty(costs)
            except:
                break
            if selected_index is None:
                #break
                debug = 0
            """
            # Print current state
            print("Current Matrix:")
            for row in costs:
                print(row)
            print("Allocated Costs:")
            for row in allocated_costs:
                print(row)
            print("\n--- Processing ---\n")"""


            if selected_index[0] == 'row':
                selected_row = selected_index[1]
                min_cost = min(costs[selected_row])
                min_cost_index = costs[selected_row].index(min_cost)
                max_supply = min(provisions[selected_row], orders[min_cost_index])
                allocated_costs[selected_row][min_cost_index] += max_supply  
                provisions[selected_row] -= max_supply
                orders[min_cost_index] -= max_supply
                costs[selected_row][min_cost_index] = float('inf')  
            else:
                selected_col = selected_index[1]
                col_values = [row[selected_col] for row in costs]
                min_cost = min(col_values)
                min_cost_index = col_values.index(min_cost)
                max_demand = min(provisions[min_cost_index], orders[selected_col])
                allocated_costs[min_cost_index][selected_col] += max_demand  
                provisions[min_cost_index] -= max_demand
                orders[selected_col] -= max_demand
                costs[min_cost_index][selected_col] = float('inf')

        # Place remaining costs in the last available cells
        for i in range(len(costs)):
            for j in range(len(costs[0])):
                if costs[i][j] != float('inf'):
                    remaining_cost = min(provisions[i], orders[j])
                    allocated_costs[i][j] += remaining_cost
                    provisions[i] -= remaining_cost
                    orders[j] -= remaining_cost
                    costs[i][j] = float('inf')
        """
        # Print final state
        print("Current Matrix:")
        for row in costs:
            print(row)
        print("Allocated Costs:")
        for row in allocated_costs:
            print(row)
        print("\n--- END Processing ---\n")"""

        table = PrettyTable()

        num_provisions = len(allocated_costs)
        num_orders = len(allocated_costs[0])

        # Ajout de la colonne pour les noms des provisions
        table.add_column("", ["S " + str(i + 1) for i in range(num_provisions)])

        # Ajout des colonnes pour chaque commande
        for i in range(num_orders):
            table.add_column("L " + str(i + 1), [allocated_costs[j][i] for j in range(num_provisions)])
        table.add_column("Provisions", self.provisions)
        total_orders = sum(self.orders)
        table.add_row(["Orders"] + self.orders + [total_orders])
        # Affichage du tableau
        table.set_style(DOUBLE_BORDER)
        print(table)

        return allocated_costs
        
    def compute_cost(self, solution):
        cost = 0
        for i in range(len(self.provisions)):
            for j in range(len(self.orders)):
                cost += solution[i][j] * self.matrix[i][j]
        return cost

    def stepping_stone(self):
        
        # TO DO : implement the stepping stone method
        # Solving algorithm : the stepping-stone method with potential.
        #     ⋆ Test whether the proposition is acyclic : we’ll use a Breadth-first algorithm. During the algorithm run, as the vertices are discovered, we check that we’re returning to a previously
        #        visited vertex and that this vertex isn’t the parent of the current vertex ; if it is, then a cycle exists. The cycle is then displayed.
        #     ⋆ Transportation maximization if a cycle has been detected. The conditions for each box are
        #        displayed. Then we display the deleted edge (possibly several) at the end of maximization.
        #     ⋆ Test whether the proposition is connected : we’ll use a Breadth-first algorithm. If it is not
        #        connected : display of all connected sub-graphs.
        #     ⋆ Modification of the graph if it is unconnected, until a non-degenerate proposition is obtained.
        #     ⋆ Calculation and display of potentials per vertex.
        #     ⋆ Display of both potential costs table and marginal costs table. Possible detection of the best
        #        improving edge.
        #     ⋆ Add this improving edge to the transport proposal, if it has been detected.

    
        solution = self.north_west_corner()
        while True:
            if is_cyclic(solution):
                for i in range(len(solution)):
                    cycle, vertices = find_cycle(get_adgency_matrix(solution), i)
                    if path:
                        break
                print("Cycle found:", path)
                path=[]
                for i in range(len(cycle)):
                    #find the max value of cycle[i]
                    if cycle[i][0] < cycle[i][1] :
                        y= cycle[i][1]-len(self.provisions)
                        x= cycle[i][0]
                    else:
                        x= cycle[i][0]-len(self.provisions)
                        y= cycle[i][1]
                    path.append((x,y))
                for i in range(len(path)):
                    if i % 2 == 0:
                        path[i] = (path[i][1], path[i][0])
                    else:
                        pass

                orders = self.orders.copy()
                provisions = self.provisions.copy()


                for i in range(len(solution)):
                    for j in range(len(solution[i])):
                        if (i,j) not in path:
                            orders[j] -= solution[i][j]
                            provisions[i] -= solution[i][j]

                path.pop()
                max_value_to_add = float('inf')
                for i in range(len(path)):
                    if i % 2 == 0:
                        if solution[path[i][0]][path[i][1]] < max_value_to_add:
                            max_value_to_add = solution[path[i][0]][path[i][1]]
                    else:
                        pass
                
                for i in range(len(path[1:])):
                    if i % 2 == 0:
                        solution[path[i][1]][path[i][0]] += max_value_to_add
                    else:
                        solution[path[i][1]][path[i][0]] -= max_value_to_add



            #cas ou la solution est acyclique mais v!=e-1    
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

            #if the graph is cyclic we need to find the cycle and delete an edge
            
            while edges != vertices - 1:
                #!we need to add an edge to the solution
                #we will find the edge with the lowest transport cost, check if we add it we will have a cycle
                min_cost = float('inf')
                min_indice = []
                for i in range(len(self.matrix)):
                    for j in range(len(self.matrix[i])):
                        if self.matrix[i][j] < min_cost and solution_graph[i][j] == 0:
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
            solution_graph = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    if solution[i][j] != 0:
                        solution_graph[i][j] = 1


            #we need to calculate the potentials
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

            solution[min_indice[0]][min_indice[1]] = 0
            solution_graph[min_indice[0]][min_indice[1]] = 1

            start_edge = min_indice[1] 
            cycle, vertices = find_cycle(get_adgency_matrix(solution_graph), start_edge)

            if not cycle:
                start_edge = min_indice[0]+ len(self.orders)
                cycle, vertices = find_cycle(get_adgency_matrix(solution_graph), start_edge) 




            path=[]
            for i in range(len(cycle)):
                #find the max value of cycle[i]
                if cycle[i][0] < cycle[i][1] :
                    y= cycle[i][1]-len(self.provisions)
                    x= cycle[i][0]
                else:
                    x= cycle[i][0]-len(self.provisions)
                    y= cycle[i][1]
                path.append((x,y))
            
            for i in range(len(path)):
                if i % 2 == 0:
                    path[i] = (path[i][1], path[i][0])
                else:
                    pass
            
            path.pop()

            orders = self.orders.copy()
            provisions = self.provisions.copy()

            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    if (i,j) not in path:
                        orders[j] -= solution[i][j]
                        provisions[i] -= solution[i][j]

            while path[0][1] != min_indice[0] or path[0][0] != min_indice[1]:
                path.append(path.pop(0))
            
            max_value_to_add = float('inf')

            for i in range(len(path)):
                if i % 2 == 1:
                    if solution[path[i][0]][path[i][1]] < max_value_to_add:
                        max_value_to_add = solution[path[i][0]][path[i][1]]
                else:
                    if max(provisions[path[i][0]], orders[path[i][1]]) < max_value_to_add:
                        max_value_to_add = max(provisions[path[i][0]], orders[path[i][1]])

            for i in range(len(path)):
                if i % 2 == 1:
                    solution[path[i][0]][path[i][1]] += max_value_to_add
                else:
                    solution[path[i][0]][path[i][1]] -= max_value_to_add






            
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
           
        if is_cyclic(solution_graph):
            print("The solution is cyclic")
            #! we need to find the cycle
            adj_matrix = get_adgency_matrix(solution_graph)
            for i in range(len(solution_graph)):
                # raise Exception("Cycle found")
                cycle, path = find_cycle(adj_matrix, i)
                if cycle:
                    print("Cycle found:", cycle)
                    print("Vertices used:", path)
                    #! we need to maximise the cycle
                    #! we need to find the edge to delete
                    min_capacity = float('inf')
                    min_edge = None
                    for edge in cycle:
                        if solution[edge[0]][edge[1]] < min_capacity:
                            min_capacity = solution[edge[0]][edge[1]]
                            min_edge = edge
                    print("Edge to delete:", min_edge)



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

        print("Stepping stone classic")
        # Création du tableau PrettyTable
        table = PrettyTable()

        num_provisions = len(solution)
        num_orders = len(solution[0])

        # Ajout de la colonne pour les noms des provisions
        table.add_column("", ["S " + str(i + 1) for i in range(num_provisions)])

        # Ajout des colonnes pour chaque commande
        for i in range(num_orders):
            table.add_column("L " + str(i + 1), [solution[j][i] for j in range(num_provisions)])
        table.add_column("Provisions", self.provisions)
        total_orders = sum(self.orders)
        table.add_row(["Orders"] + self.orders + [total_orders])
        # Affichage du tableau
        table.set_style(DOUBLE_BORDER)
        print(table)
        return solution
    
    def compute_potentials(self, solution):
        string = ""
        system = []
        for i in range(len(solution)):
            for j in range(len(solution[i])):
                if solution[i][j] != 0:
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
        return potentials_row, potentials_col
    
    def compute_marginals_costs(self, potentials_row, potentials_col):
        potential_costs = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
        for i in range(len(self.provisions)):
            for j in range(len(self.orders)):
                potential_costs[i][j] = potentials_row[i] - potentials_col[j]

        marginal_costs = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
        for i in range(len(self.provisions)):
            for j in range(len(self.orders)):
                marginal_costs[i][j] = self.matrix[i][j] - potential_costs[i][j]
        return marginal_costs
    
    def stepping_stone_working(self):

        solution = self.ballas_hammer()
        #print("Degenerée ? : ",is_degenerate(solution))
        self.fix_E_V(solution)
        #print("tablop")
        for i in solution: print(i)
        #fix_cycle(self,solution)

        while True:



            solution_graph = [[0 for i in range(len(self.orders))] for j in range(len(self.provisions))]
            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    if solution[i][j] != 0:
                        solution_graph[i][j] = 1

            #we need to calculate the potentials

            potentials_row,potentials_col = self.compute_potentials(solution_graph) 
            
            #compute the marginal costs

            marginal_costs = self.compute_marginals_costs(potentials_row, potentials_col)            

            #we find the edge with the lowest marginal cost
            
            min_cost, min_indice = find_minimum_in_matrix(marginal_costs)

            if min_cost >= 0:
                break

            solution_graph[min_indice[0]][min_indice[1]] = 1

            start_edge = min_indice[1] 
            cycle, vertices = find_cycle(get_adgency_matrix(solution_graph), start_edge)
            print("Cycle :",cycle)
            
            if not cycle:
                start_edge = min_indice[0]+ len(self.orders)
                cycle, vertices = find_cycle(get_adgency_matrix(solution_graph), start_edge)
            path=[]
            for i in range(len(cycle)):
                #find the max value of cycle[i]
                if cycle[i][0] < cycle[i][1] :
                    y= cycle[i][1]-len(self.provisions)
                    x= cycle[i][0]
                else:
                    x= cycle[i][0]-len(self.provisions)
                    y= cycle[i][1]
                path.append((x,y))
            
            for i in range(len(path)):
                if i % 2 == 0:
                    path[i] = (path[i][1], path[i][0])
                else:
                    pass
            
            path.pop()
            orders = self.orders.copy()
            provisions = self.provisions.copy()
            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    if (i,j) not in path:
                        orders[j] -= solution[i][j]
                        provisions[i] -= solution[i][j]

            while path[0][1] != min_indice[0] or path[0][0] != min_indice[1]:
                path.append(path.pop(0))
            
            max_value_to_add = float('inf')
            for i in range(len(path)):
                if i % 2 == 1:
                    if solution[path[i][0]][path[i][1]] < max_value_to_add:
                        max_value_to_add = solution[path[i][0]][path[i][1]]
                else:
                    if max(provisions[path[i][0]], orders[path[i][1]]) < max_value_to_add:
                        max_value_to_add = max(provisions[path[i][0]], orders[path[i][1]])
            for i in range(len(path)):
                if i % 2 == 0:
                    solution[path[i][0]][path[i][1]] += max_value_to_add
                else:
                    solution[path[i][0]][path[i][1]] -= max_value_to_add

            #print("Stepping stone working")
            # Création du tableau PrettyTable
            table = PrettyTable()

            num_provisions = len(solution)
            num_orders = len(solution[0])

            # Ajout de la colonne pour les noms des provisions
            table.add_column("", ["S " + str(i + 1) for i in range(num_provisions)])

            # Ajout des colonnes pour chaque commande
            for i in range(num_orders):
                table.add_column("L " + str(i + 1), [solution[j][i] for j in range(num_provisions)])
            table.add_column("Provisions", self.provisions)
            total_orders = sum(self.orders)
            table.add_row(["Orders"] + self.orders + [total_orders])
            # Affichage du tableau
            table.set_style(DOUBLE_BORDER)
            print(table)



    def __str__(self): #TODO : implement the __str__ method -> @Mathieu fait nous des beaux tableaux 
        """
        # cost cost cost cost cost prov
        # cost cost cost cost cost prov
        # cost cost cost cost cost prov
        # cost cost cost cost cost prov
        # ord ord ord ord ord
        """
        string = ""
        for i in range(len(self.matrix)):
            string += " ".join(str(x) for x in self.matrix[i]) + " " + str(self.provisions[i]) + "\n"
        string += " ".join(str(x) for x in self.orders)
        #print(string)
        



        


        return f"Orders : {self.orders}\nProvisions : {self.provisions}\nMatrix : {self.matrix}"

class TextStyle:
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def complexity_study():
    Tnw = []
    Tbh = []
    SSnw = []
    SSbh = []
    n = [10, 40, 100, 400, 1000, 4000, 10000]
    n_values = ['10', '40', '10^2', '4*10^2', '10^3', '4*10^3', '10^4']
    for i in n:
        for j in range(100):
            matrice = generate(i)
            start = time.time()
            matrice.north_west_corner # matrixNW = NW(matrix)
            end = time.time()
            Tnw.append((i,end-start))
            start = time.time()
            matrice.north_west_corner # SS de matrixNW
            end = time.time()
            SSnw.append((i,end-start))
            start = time.time()
            matrice.ballas_hammer # matrixBH = BH(matrix)
            end = time.time()
            Tbh.append((i,end-start))
            start = time.time()
            matrice.ballas_hammer # SS de matrixBH
            end = time.time()
            SSbh.append((i,end-start))
            NW = Tnw + SSnw
            BH = Tbh + SSbh
    scatter_plots = [(Tnw, "Tnw"), (Tbh, "Tbh"), (SSnw, "SSnw"), (SSbh, "SSbh"), (NW, "Total North-West"), (BH, "Total Ballass-Hammer")]

    for idx, (data, label) in enumerate(scatter_plots):
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure for each plot
        ax.scatter(*zip(*data), s=100)
        ax.set_title(label)
        ax.set_xlabel('n')
        ax.set_ylabel('Time (s)')
        ax.set_xticks(n)
        ax.set_xticklabels(n_values)
        plt.tight_layout()
        plt.show()


def generate(n):
    matrice = transportation_problem()
    values = np.random.randint(1, 100, size=n)
    np.random.shuffle(values)
    matrice.provisions = values.tolist()
    np.random.shuffle(values)
    matrice.orders = values.tolist()
    matrice.matrix = np.random.randint(1, 100, size=(n, n))
    return matrice



def menu():
    # complexity_study()
    while True:
        print(TextStyle.RED + TextStyle.BOLD + "\nWelcome to our Operations Research project!" + TextStyle.END)
        problem_number = int(input("Enter the transportation problem number to be processed (1-12) or 13 to generate a random matrice: "))
        while not (1 <= problem_number <= 13):
            print("Invalid input. Please enter an integer between 1 and 12.")
            problem_number =  int(input("Enter the transportation problem number to be processed (1-12): "))
        if problem_number == 13:   
            test = transportation_problem()
            test.init_random(3,3)
            print("test :\n",test.matrix)

        else: 
            test = transportation_problem('files/problem_'+str(problem_number)+'.txt')
        print(TextStyle.RED + TextStyle.BOLD + "\nCost Matrix : " + TextStyle.END)
        # Création du tableau PrettyTable
        table = PrettyTable()

        num_provisions = len(test.matrix)
        num_orders = len(test.matrix[0])

        # Ajout de la colonne pour les noms des provisions
        table.add_column("", ["S " + str(i + 1) for i in range(num_provisions)])

        # Ajout des colonnes pour chaque commande
        for i in range(num_orders):
            table.add_column("L " + str(i + 1), [test.matrix[j][i] for j in range(num_provisions)])
        table.add_column("Provisions", test.provisions)
        total_orders = sum(test.orders)
        table.add_row(["Orders"] + test.orders + [total_orders])
        # Affichage du tableau
        table.set_style(DOUBLE_BORDER)
        print(table)

        init_prop = int(input("Choose the initial proposal you want :\n1. North-West\n2.Ballas-Hammer\n"))
        while not (1 <= init_prop <= 2):
            print("Invalid input. Please enter 1 or 2.")
            init_prop = int(input("Choose the initial proposal you want:\n1. North-West\n2.Ballas-Hammer\n"))
        print("\nTransportation Proposal for", end=" ")
        if init_prop == 1:
            print(TextStyle.RED + TextStyle.BOLD + "\nNorth West :" + TextStyle.END)
            matrice = test.north_west_corner()
        elif init_prop == 2:
            print(TextStyle.RED + TextStyle.BOLD + "\nBallas-Hammer :" + TextStyle.END)
            matrice = test.ballas_hammer()
        print("Now running the Stepping-Stone method:")

        steppingStone(matrice,test.matrix,[0 for i in range (len(matrice[0]))],[0 for i in range (len(matrice))],test.orders,test.provisions)

        choice = int(input("Do you want to choose another problem number?\n 1. YES\n 2. NO\n"))
        if choice != 1:
            print("Exiting...")
            break

menu()
