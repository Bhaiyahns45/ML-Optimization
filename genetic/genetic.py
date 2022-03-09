import numpy as np



def initialization(M, N):

    population = []

    for i in range(M):
        #     population.append([np.random.randint(0,2) , np.random.randint(0,2)])
        population.append([np.random.randint(0, 2) for x in range(N)])

    return population


def fitness_fun(X):

    X = [x**2 for x in X]

    fitness_value = sum(X)

    return fitness_value


def rolette_wheel(p):

    cumsum = np.cumsum(p)

    r = np.random.rand()

    next_node = np.where(r <= cumsum)

    parent1_index = next_node[0][0]

    parent2_index = parent1_index

    while parent1_index == parent2_index:

        r = np.random.rand()

        next_node = np.where(r <= cumsum)

        parent2_index = next_node[0][0]

    return parent1_index, parent2_index


def selecting_two_parent(fitness_value):

    f = fitness_value

    if any(i < 0 for i in f):
        a = 1
        b = abs(min(f))

        print("------")
        print(b)

        scaled_fitness = [x+1+abs(min(f)) for x in f]

        normalized_fitness = scaled_fitness / np.sum(scaled_fitness)

    else:

        normalized_fitness = f / np.sum(f)

    parent1_index, parent2_index = rolette_wheel(normalized_fitness)

    return parent1_index, parent2_index


def crossover(parent1, parent2, crossover_name, pc):

    if crossover_name == "single":

        l = len(parent1)
        cross_p1 = np.random.randint(1, l)

        part1 = [parent1[i] for i in range(cross_p1)]
        part2 = [parent2[i] for i in range(cross_p1, l)]

        child1 = part1+part2

        part1 = [parent2[i] for i in range(cross_p1)]
        part2 = [parent1[i] for i in range(cross_p1, l)]

        child2 = part1+part2

    elif crossover_name == "double":

        l = len(parent1)
        cross_p1 = np.random.randint(1, l)

        cross_p2 = cross_p1

        while cross_p2 == cross_p1:
            cross_p2 = np.random.randint(1, l)

        if cross_p1 > cross_p2:
            cross_p1, cross_p2 = cross_p2, cross_p1

        part1 = [parent1[i] for i in range(cross_p1)]
        part2 = [parent2[i] for i in range(cross_p1, cross_p2)]
        part3 = [parent1[i] for i in range(cross_p2, l)]

        child1 = part1+part2+part3

        part1 = [parent2[i] for i in range(cross_p1)]
        part2 = [parent1[i] for i in range(cross_p1, cross_p2)]
        part3 = [parent2[i] for i in range(cross_p2, l)]

        child2 = part1+part2+part3

    else:
        print("wrong input")

    r1 = np.random.rand()

    if r1 <= pc:
        child1 = child1
    else:
        child1 = parent1

    r2 = np.random.rand()

    if r2 <= pc:
        child2 = child2
    else:
        child2 = parent2

    return child1, child2


def mutation(child, pm):

    l = len(child)

    for i in range(l):
        r = np.random.rand()

        if r <= pm:
            child[i] = int(not(child[i]))

    return child


def elitism(er, fitness_value, population, new_population):

    elite_no = round(len(fitness_value)*er)

    index = list(range(len(fitness_value)))

    l=len(fitness_value)

    index.sort(key=lambda i: fitness_value[i], reverse=True)

    new_population2 = [population[index[i]] for i in range(elite_no)]

    for j in range(elite_no, l):
        new_population2.append(new_population[j])

    return new_population2


M = 20
N = 10
max_gen = 10
pc = 0.95
pm = 0.001
er = 0.2

population = initialization(M, N)


print(population)

# main loop

for g in range(max_gen):

    
    fitness_value = [fitness_fun(i) for i in population]

    new_population = []

    for i in range(0, M, 2):

        parent1_index, parent2_index = selecting_two_parent(fitness_value)

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

 

        child1, child2 = crossover(parent1, parent2, "single", pc)

     
        mutated_child1 = mutation(child1, pm)
        mutated_child2 = mutation(child2, pm)

        new_population.append(mutated_child1)
        new_population.append(mutated_child2)

        # new_population.append(child1)
        # new_population.append(child2)


    new_population = elitism(er, fitness_value, population, new_population)

    # print(new_population)

    population = new_population

    print("\nGeneration #", g+1)
    fitness_value = [fitness_fun(i) for i in population]
    print(fitness_value)

    print(max(fitness_value))

    


fitness_value = [fitness_fun(i) for i in population]

print()
print(fitness_value)


best_gene = population[fitness_value.index(max(fitness_value))]
best_fitness = max(fitness_value)

print("\nBest chromosome")
print(" Gene = ",best_gene)
print(" Fitness = ",best_fitness)
# print(new_population)
