from main import transportation_problem
import time

with open('traces/int2-2_traces{}.txt'.format(1), 'w') as file:
    graph = transportation_problem('files/problem_{}.txt'.format(1))
    str_graph = '------------------- Transportation Problem : {} -------------------\n'.format(1)
    str_graph += '\nGraph : \n'
    str_graph += str(graph)
    str_graph += '\n\n'
    str_graph += '------------------- North-West Corner Method -------------------\n'
    durée = time.time_ns()
    soluce = graph.north_west_corner()
    str_graph += 'Time : {} \n\n'.format( time.time_ns() - durée)
    for i in range(len(soluce)):
        for j in range(len(soluce[i])):
            str_graph += '{} '.format(soluce[i][j])
        str_graph += '|{}\n'.format(graph.provisions[i])
    for i in range(len(graph.orders)):
        str_graph += '{} '.format(graph.orders[i])
    
    
    str_graph += '\n\n'
    str_graph += '------------------- Ballas-Hamer method -------------------\n'
    durée = time.time_ns()
    soluce = graph.ballas_hammer()
    str_graph += 'Time : {} \n\n'.format( time.time_ns() - durée)
    for i in range(len(soluce)):
        for j in range(len(soluce[i])):
            str_graph += '{} '.format(soluce[i][j])
        str_graph += '|{}\n'.format(graph.provisions[i])
    for i in range(len(graph.orders)):
        str_graph += '{} '.format(graph.orders[i])
    str_graph += '\n\n'
    str_graph += '------------------- stepping stone -------------------\n'
    durée = time.time_ns()
    soluce = graph.north_west_corner()
    str_graph += 'Time : {} \n\n'.format( time.time_ns() - durée)
    for i in range(len(soluce)):
        for j in range(len(soluce[i])):
            str_graph += '{} '.format(soluce[i][j])
        str_graph += '|{}\n'.format(graph.provisions[i])
    for i in range(len(graph.orders)):
        str_graph += '{} '.format(graph.orders[i])
    str_graph += '\n\n'

    file.write(str_graph)

def writeGraph(index):
    with open('traces/int2-2_traces{}.txt'.format(index), 'w') as file:
        graph = transportation_problem('files/problem_{}.txt'.format(index))
        str_graph = '------------------- Transportation Problem : {} -------------------\n'.format(index)
        str_graph += '\nGraph : \n'
        str_graph += str(graph)
        str_graph += '\n\n'
        str_graph += '------------------- North-West Corner Method -------------------\n'
        durée = time.time_ns()
        soluce = graph.north_west_corner()
        str_graph += 'Time : {} \n\n'.format( time.time_ns() - durée)
        for i in range(len(soluce)):
            for j in range(len(soluce[i])):
                str_graph += '{} '.format(soluce[i][j])
            str_graph += '|{}\n'.format(graph.provisions[i])
        for i in range(len(graph.orders)):
            str_graph += '{} '.format(graph.orders[i])
    
    
        str_graph += '\n\n'
        str_graph += '------------------- Ballas-Hamer method -------------------\n'
        durée = time.time_ns()
        soluce = graph.ballas_hammer()
        str_graph += 'Time : {} \n\n'.format( time.time_ns() - durée)
        for i in range(len(soluce)):
            for j in range(len(soluce[i])):
                str_graph += '{} '.format(soluce[i][j])
            str_graph += '|{}\n'.format(graph.provisions[i])
        for i in range(len(graph.orders)):
            str_graph += '{} '.format(graph.orders[i])
        str_graph += '\n\n'
        str_graph += '------------------- stepping stone -------------------\n'
        durée = time.time_ns()
        soluce = graph.north_west_corner()
        str_graph += 'Time : {} \n\n'.format( time.time_ns() - durée)
        for i in range(len(soluce)):
            for j in range(len(soluce[i])):
                str_graph += '{} '.format(soluce[i][j])
            str_graph += '|{}\n'.format(graph.provisions[i])
        for i in range(len(graph.orders)):
            str_graph += '{} '.format(graph.orders[i])
        str_graph += '\n\n'

        file.write(str_graph)

writeGraph(12)

def test_n_matrix_problems(size, number):
    string=""
    for i in range(number):
        test = transportation_problem(x=size, y=size)
        time1 = time.time_ns()
        test.north_west_corner()
        time2 = time.time_ns()
        string += "{}\t".format(time2-time1)
        time1 = time.time_ns()
        test.ballas_hammer()
        time2 = time.time_ns()
        string += "{}\t".format(time2-time1)
        string += "\n"
    
    with open('data/duration_{}_size.txt'.format(size), 'w') as file:
        file.write(string)

test_n_matrix_problems(10, 10)

