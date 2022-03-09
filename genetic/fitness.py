import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def initialization(M, N):

    population = []

    for i in range(M):
        #     population.append([np.random.randint(0,2) , np.random.randint(0,2)])
        population.append([np.random.randint(-100, 101) for x in range(N)])

    return population


def fitness_fun(X):

    X = [x**2 for x in X]

    fitness_value = sum(X)

    fitness_value = -1*fitness_value

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

        scaled_fitness = [x+1+b for x in f]

        normalized_fitness = scaled_fitness / np.sum(scaled_fitness)

    else:

        normalized_fitness = f / np.sum(f)

    parent1_index, parent2_index = rolette_wheel(normalized_fitness)

    return parent1_index, parent2_index


def crossover(parent1, parent2, pc):

    l = len(parent1)

    child1 = [0]*l
    child2 = [0]*l

    for j in range(l):

        beta = np.random.rand()

        x = parent1[j]
        y = parent2[j]

        child1[j] = (beta*x) + (1-beta)*y

        child2[j] = (1-beta)*x + (beta)*y

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
            child[i] = np.random.randint(-100, 101)  # mutate the i-th gene

    return child


def elitism(er, fitness_value, population, new_population):

    elite_no = round(len(fitness_value)*er)

    index = list(range(len(fitness_value)))

    index.sort(key=lambda i: fitness_value[i], reverse=True)

    l = len(fitness_value)

    new_population2 = [population[index[i]] for i in range(elite_no)]

    # new_population2 = [new_population[index[i]] for i in range(elite_no, l)]

    for j in range(elite_no, l):
        new_population2.append(new_population[j])

    # for i in range(elite_no):

    #     # print(index[i])
    #     # print(population[index[i]])
    #     new_population[i] = population[index[i]]

    return new_population2


# parameters
M = 20
N = 2
max_gen = 20
pc = 0.95
pm = 0.001
er = 0.2

population = initialization(M, N)


# print(population)

fitness_value = [fitness_fun(i) for i in population]
print(fitness_value)

print(max(fitness_value))

# main loop

all_fintess=[]

for g in range(max_gen):


    fitness_value = [fitness_fun(i) for i in population]

    new_population = []

    for i in range(0, M, 2):

        parent1_index, parent2_index = selecting_two_parent(fitness_value)

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        child1, child2 = crossover(parent1, parent2, pc)

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
    # print(fitness_value)

    all_fintess.append(max(fitness_value))
    print(max(fitness_value))


plt.title("Convergence of fitness value", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("Iteration")
plt.ylabel("fitness value")
plt.plot(range(max_gen), all_fintess)
plt.xticks(np.arange(1, max_gen, 1.0))
plt.show()

fitness_value = [fitness_fun(i) for i in population]

print()
# print(fitness_value)


best_gene = population[fitness_value.index(max(fitness_value))]
best_fitness = max(fitness_value)

print("\nBest chromosome")
print(" Gene = ", best_gene)
print(" Fitness = ", best_fitness)
# print(new_population)
