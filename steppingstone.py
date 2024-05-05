import copy

import networkx as nx

from prettytable import PrettyTable,DOUBLE_BORDER
import math
def printingMarginal(marginal_costs):
    for i in range(len(marginal_costs)):
        for j in range(len(marginal_costs[i])):
            marginal_costs[i][j] = math.sqrt(marginal_costs[i][j]**2)
    
    table = PrettyTable()

    table.field_names = [""] + [f"Column {i+1}" for i in range(len(marginal_costs[0]))]

    for i, row in enumerate(marginal_costs):
        table.add_row([f"Row {i+1}"] + row)

    print(table)

def Costculation(cost_matrice,datalist,p=True):
    cal=0
    for i in range(len(cost_matrice)):
        for j in range(len(cost_matrice[i])):
            cal+=cost_matrice[i][j]*datalist[i][j]
    if p:
        print(cal)
    return cal

def display_graph(datalist, p=True):
    G = nx.Graph()

    cols = len(datalist[0])
    for j in range(cols):
        G.add_node(f"C{j + 1}", pos=(j, 1))

    rows = len(datalist)
    for i in range(rows):
        G.add_node(f"P{i + 1}", pos=(i, 0))

    for i in range(rows):
        for j in range(cols):
            if datalist[i][j] != 0:
                G.add_edge(f"P{i + 1}", f"C{j + 1}", weight=datalist[i][j])

    return G

def testcircular(g,p=False):
    try:
        cycle=nx.find_cycle(g, orientation='original')
        string=""
        for i in cycle:
            string=string + " ==> "+ i[0]
        a=cycle
    except:
        a=1
    return a


def can_reach_all_nodes(graph):
    num_nodes = len(graph.nodes())

    # Go through every node of a graph
    for node in graph.nodes():
        visited = set()  # Every visited node

        def dfs(current_node):
            visited.add(current_node)

            # Go through every neighbour of current node
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    dfs(neighbor)

        dfs(node)

        # Check that all nodes have been visited
        if len(visited) != num_nodes:
            return False

    return True

def printing(datalist, order, provision):
    datalist_copy = copy.deepcopy(datalist)
    order_copy = copy.deepcopy(order)
    provision_copy = copy.deepcopy(provision)

    headers = ["-"]
    for i in range(len(datalist_copy[0])):
        headers.append(f"C{i+1}")
    headers.append("Provision")

    for i in range(len(datalist_copy)):
        datalist_copy[i].insert(0, f"P{i+1}")
        datalist_copy[i].append(provision_copy[i])
    order_copy.insert(0, "Order")
    order_copy.append("-")
    datalist_copy.append(order_copy)

    table = PrettyTable()

    table.field_names = headers


    for row in datalist_copy:
        table.add_row(row)

    table.set_style(DOUBLE_BORDER)
    print(table)

def fixCycle(test, cost_matrice, graph, order, provision, upgrade):
    transport_copy1 = copy.deepcopy(cost_matrice)

    do = None
    mini = None
    # Finds the minimum edge in the cycle
    for data in test:
        if "P" in data[1]:
            i = int(data[1][1:]) - 1
            j = int(data[0][1:]) - 1
            if mini == None:
                do = "P"
                mini = transport_copy1[i][j]
            if transport_copy1[i][j] < mini:
                do = "P"
                mini = transport_copy1[i][j]
        else:
            i = int(data[0][1:]) - 1
            j = int(data[1][1:]) - 1
            if mini == None:
                do = "C"
                mini = transport_copy1[i][j]
            if transport_copy1[i][j] < mini:
                do = "C"
                mini = transport_copy1[i][j]
    quantity = None

    # Finds the minimal quantity among edges that will decrease
    for data in test:
        if do == "C":
            i = int(data[1][1:]) - 1
            j = int(data[0][1:]) - 1
            if "P" in data[1]:
                if quantity == None:
                    quantity = transport_copy1[i][j]
                if quantity > transport_copy1[i][j]:
                    quantity = transport_copy1[i][j]
        else:
            i = int(data[0][1:]) - 1
            j = int(data[1][1:]) - 1
            if "C" in data[1]:
                if quantity == None:
                    quantity = transport_copy1[i][j]
                if quantity > transport_copy1[i][j]:
                    quantity = transport_copy1[i][j]

    # Deletes an edge with quantity = 0 that is part of the edges that should decrease to fix the cycle
    if quantity == 0:
        for data in test:
            if "C" in data[0]:
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
            else:
                i = int(data[0][1:]) - 1
                j = int(data[1][1:]) - 1
            if transport_copy1[i][j] == 0:
                if upgrade[1] != i or upgrade[0] != j:
                    graph.remove_edge(data[0], data[1])

    # Shifts the values in the cycle
    else:
        for data in test:
            if do == "C":
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
                if "P" in data[1]:
                    transport_copy1[i][j] -= quantity
                else:
                    transport_copy1[j][i] += quantity
            else:
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
                if "P" in data[1]:
                    transport_copy1[i][j] += quantity
                else:
                    transport_copy1[j][i] -= quantity
        cost_matrice = transport_copy1

        # Removes edges that are now 0 in quantity
        for data in test:

            if "P" in data[1]:
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
                if transport_copy1[i][j] == 0:
                    graph.remove_edge(data[0], data[1])
            else:

                i = int(data[0][1:]) - 1
                j = int(data[1][1:]) - 1
                if transport_copy1[i][j] == 0:
                    graph.remove_edge(data[0], data[1])
    return (cost_matrice, graph)

def testContinuity(g,cost):
    while not nx.is_connected(g):
        connected_components = list(nx.connected_components(g))
        i1=[]
        j1=[]
        i2=[]
        j2=[]
        for i in connected_components[0]:
            if "C" in i:
                j1.append(int(i[1:])-1)
            else:
                i1.append(int(i[1:])-1)
        for i in connected_components[1]:
            if "C" in i:
                j2.append(int(i[1:])-1)
            else:
                i2.append(int(i[1:])-1)
        mini=None
        for i in i1:
            for j in j2:
                if mini==None:
                    doi=i
                    doj=j
                    mini=cost[i][j]
                elif mini>cost[i][j]:
                    doi=i
                    doj=j
                    mini=cost[i][j]
        for i in i2:
            for j in j1:
                if mini==None:
                    doi=i
                    doj=j
                    mini=cost[i][j]
                elif mini>cost[i][j]:
                    doi=i
                    doj=j
                    mini=cost[i][j]
        g.add_edge("P"+str(doi+1),"C"+str(doj+1))
    return g


def calc_potentials_cout(table, potentials):
    potential_costs = []
    tableC = []
    tableP = []

    for i in range(len(table)):
        tableP.append(i)
    for i in range(len(table[0])):
        tableC.append(i)
    for i in range(len(tableP)):
        temp = []
        for j in range(len(tableC)):
            temp.append(potentials["P" + str(i + 1)] - potentials["C" + str(j + 1)])
        potential_costs.append(temp)
    return potential_costs


# Uses cost and potential cost tables to create marginal costs table (Costs - potential costs)
def calculate_marginal_costs(costs, potential_costs):
    marginal_costs = []
    for i in range(len(potential_costs)):
        temp = []
        for j in range(len(potential_costs[i])):
            temp.append(costs[i][j] - potential_costs[i][j])
        marginal_costs.append(temp)
    return (marginal_costs)


# detect the greatest negative number among marginal costs
def detect_amelioration(marginal_costs, cost_matrice):
    maxi = 0
    for i in marginal_costs:
        for j in i:
            if cost_matrice[marginal_costs.index(i)][i.index(j)] == 0:
                if maxi > j:
                    maxi = j
                    imaxi = marginal_costs.index(i)
                    jmaxi = i.index(j)
    if maxi != 0:
        return (jmaxi, imaxi)

def afficher_potentiel(potentials):
    print("Potentiels par sommet :")
    for node, potential in potentials.items():
        print(f"Sommet {node}: {potential}")
    print()


def calc_potentials(graph, costs):
    num_nodes = len(graph.nodes())
    potentials = {}
    for i in graph.nodes:
        potentials[i] = None

    start_node = list(graph.nodes())[0]
    potentials[start_node] = 0
    todo = copy.deepcopy(graph.edges)
    while None in potentials.values():
        for i in todo:

            if potentials[i[0]] != None and potentials[i[1]] == None:
                if "C" in i[0]:
                    ical = int(i[1][1:]) - 1
                    jcal = int(i[0][1:]) - 1
                    potentials[i[1]] = costs[ical][jcal] + potentials[i[0]]
                else:
                    ical = int(i[0][1:]) - 1
                    jcal = int(i[1][1:]) - 1
                    potentials[i[1]] = potentials[i[0]] - costs[ical][jcal]
            elif potentials[i[1]] != None and potentials[i[0]] == None:
                if "C" in i[0]:
                    ical = int(i[1][1:]) - 1
                    jcal = int(i[0][1:]) - 1
                    potentials[i[0]] = -costs[ical][jcal] + potentials[i[1]]
                else:
                    ical = int(i[0][1:]) - 1
                    jcal = int(i[1][1:]) - 1
                    potentials[i[0]] = potentials[i[1]] - costs[ical][jcal]
    return potentials

def steppingStone(cost_matrice,datalist,transportorder,transportprovision,order,provision):
    a = 0
    g = display_graph(cost_matrice)
    print("The graph is:\n", g)
    best_improvement = None
    err = 0
    precost = Costculation(cost_matrice, datalist, p=False)
    while a == 0:

        test = testcircular(g, p=True)
        if test != 1:
            print("There is a cycle: ", test[0][0], end="")
            for edge in test:
                print(" ==> ", edge[1], end="")
            print("\n")
        if can_reach_all_nodes(g) == False:
            print("The graph is degenerate")
            sub_graphs = list(nx.connected_components(g))
            for graph in sub_graphs:
                print(f"Sub graph: {graph}")
        while test != 1 or can_reach_all_nodes(g) == False:

            if test != 1:
                (cost_matrice, g) = fixCycle(test, cost_matrice, g, order, provision, best_improvement)
            if can_reach_all_nodes(g) == False:
                g = testContinuity(g, datalist)

            test = testcircular(g)
        potentials = calc_potentials(g, datalist)

        afficher_potentiel(potentials)
        potential_costs = calc_potentials_cout(cost_matrice, potentials)
        marginal_costs = calculate_marginal_costs(datalist, potential_costs)
        print('-- The potential costs is: ')
        table = PrettyTable()
        for row in potential_costs:
            table.add_row(row)

        table.set_style(DOUBLE_BORDER)
        print(table)
        print('-- The marginal costs is: ')
        table = PrettyTable()
        for row in marginal_costs:
            table.add_row(row)

        table.set_style(DOUBLE_BORDER)
        print(table)
        best_improvement = detect_amelioration(marginal_costs, cost_matrice)
        if best_improvement is not None:

            if (err == 10):
                printingMarginal(marginal_costs)
                print("No improvement detected.")
                a = 1
                cost = 0
                for i in range(len(cost_matrice)):
                    for j in range(len(cost_matrice[i])):
                        cost += cost_matrice[i][j] * datalist[i][j]
                print("Total cost: ", cost)
                break
            else:
                print("Improvement detected :", best_improvement)
                print("The current cost is")
                postcost = Costculation(cost_matrice, datalist)
                g.add_edge("P" + str(best_improvement[1] + 1), "C" + str(best_improvement[0] + 1))
                if precost == postcost:
                    err += 1
                else:
                    err = 1
                precost = postcost

            # printing(cost_matrice,transportorder,transportprovision)
        else:
            print("No improvement detected.")
            a = 1
            cost = 0
            for i in range(len(cost_matrice)):
                for j in range(len(cost_matrice[i])):
                    cost += cost_matrice[i][j] * datalist[i][j]
            printing(cost_matrice, order, provision)
            print("Total cost: ", cost)