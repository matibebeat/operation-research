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
        #self.ballas_hammer()
        



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

    def ballas_hammer(self): #TODO : implement the ballas hammer algorithm





        #!important : Not working but the beginning of the implementation is here
        orders = self.orders.copy()
        provisions = self.provisions.copy()
        solution = [[None for i in range(len(orders))] for j in range(len(provisions))]
        orders_penalty = [0 for i in range(len(orders))]
        provisions_penalty = [0 for i in range(len(provisions))]
        while True:
            for i in range(len(provisions_penalty)):
                if provisions_penalty[i] != None:
                    # find the 2 smallest values in the row
                    row = self.matrix[i].copy()
                    row.sort()
                    provisions_penalty[i] = row[1]- row[0]

            for i in range(len(orders_penalty)):
                if orders_penalty[i] != None:
                
                    # find the 2 smallest values in the column
                    column = [row[i] for row in self.matrix.copy()]
                    column.sort()
                    orders_penalty[i] = column[1] - column[0]   
            # find the biggests penalty
            penalty_to_be_solved = [ ]
            temp = provisions_penalty.copy()
            if None in temp:
                temp.remove(None)
            max_provision_penalty = max(temp)
            for i in range(len(provisions_penalty)):
                if provisions_penalty[i] == max_provision_penalty:
                    penalty_to_be_solved.append([True, i, max_provision_penalty])
            temp = orders_penalty.copy()
            if None in temp:
                temp.remove(None)
            max_order_penalty = max(temp)
            for i in range(len(orders_penalty)):
                if orders_penalty[i] == max_order_penalty:
                    penalty_to_be_solved.append([False, i, max_order_penalty])
            print(penalty_to_be_solved)
            #find the true biggest penalty
            true_max_penalty = 0
            for i in range(len(penalty_to_be_solved)):
                if penalty_to_be_solved[i][2] > true_max_penalty:
                    true_max_penalty = penalty_to_be_solved[i][2]
            true_penalty_to_be_solved = []
            for i in range(len(penalty_to_be_solved)):
                if penalty_to_be_solved[i][2] == true_max_penalty:
                    true_penalty_to_be_solved.append(penalty_to_be_solved[i])
            print(true_penalty_to_be_solved)

            #find the smallests transportation costs 
            min_cost = None
            for i in range(len(provisions)):
                for j in range(len(orders)):
                    if min_cost == None:
                        min_cost = self.matrix[i][j]
                        min_i = i
                        min_j = j
                    elif self.matrix[i][j] < min_cost:
                        min_cost = self.matrix[i][j]
                        min_i = i
                        min_j = j
            #find all the cells with the same cost and are in the true penalties

            min_costs = []
            for elem in true_penalty_to_be_solved:
                if elem[0]:
                   for i in range(len(self.matrix.copy())):
                        if self.matrix[elem[1]][i] == min_cost:
                            #find the amount to be transported
                            if provisions[elem[1]] < orders[i]:
                                min_costs.append([elem[1], i, provisions[elem[1]]])
                            else:
                                min_costs.append([elem[1], i, orders[i]])
                            
                else:
                    for i in range(len(self.matrix.copy())):
                        if self.matrix[i][elem[1]] == min_cost:
                            #find the amount to be transported
                            if provisions[i] < orders[elem[1]]:
                                min_costs.append([i, elem[1], provisions[i]])
                            else:
                                min_costs.append([i, elem[1], orders[elem[1]]])
            #find the biggest amount to be transported #!important until here the code is working 
            max_transport = 0
            max_transport_x = 0
            max_transport_y = 0
            for elem in min_costs:
                if elem[2] > max_transport:
                    max_transport = elem[2]
                    max_transport_x = elem[0]
                    max_transport_y = elem[1]
            #update the solution by the biggest amount to be transported
            solution[max_transport_x][max_transport_y] = max_transport
            #update the orders and provisions
            orders[max_transport_y] -= max_transport
            provisions[max_transport_x] -= max_transport
            #update the penalties
            orders_penalty[max_transport_y] = None
            provisions_penalty[max_transport_x] = None


            print(min_costs)
        
        print(self.matrix)
        print(provisions_penalty)
        print(orders_penalty)

        return solution
        

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
