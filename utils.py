def solve_2nd_order_system(liste , answers = {}):
    """
    Solve a 2nd order system of equations.
    take as parameter a list of 2nd order equations in the form of a str: "x1-x2=solution"
    the solution of all the equations are known and the system is solvable and the operator is always a minus
    exemple: ["x1-x2=10", "x3-x4=15"]
    return a dict with the solution of the system : {"x1": value, "x2": value}
    :param list: list of equations
    :return: dict
    """
    #loop over the list of equations
    for eq in liste:
        #split the equation into the two variables
        eq = eq.split("=")
        solution = eq[1]
        #split the variables and the sign
        x1 , x2 = eq[0].split("-")
        #check if x1 is a key in the dict
        if x1 in answers:
            if not x2 in answers:
                answers[x2] = answers[x1] - int(solution)
                for elem in liste :
                    #si x2 est dans elem (str) remplace x2 par la valeur de x2
                    elem = elem.replace(x2, str(answers[x2]))
                    
                answer = solve_2nd_order_system(liste , answers)
        else:
            if x2 in answers:
                answers[x1] = answers[x2] + int(solution)
                for elem in liste :
                    #si x1 est dans elem (str) remplace x1 par la valeur de x1
                    elem = elem.replace(x1, str(answers[x1]))
                answer = solve_2nd_order_system(liste , answers)
    return answers
                

print(solve_2nd_order_system(["e1-c2=10", "c2-e4=15" ], {"e4": 5}))

            

        
