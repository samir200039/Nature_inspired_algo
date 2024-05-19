import random

def random_chromosome(size):
    return [random.randint(0, size - 1) for _ in range(size)]

def fitness(chromosome, maxFitness):
    horizontal_collisions = sum([chromosome.count(queen) - 1 for queen in chromosome]) / 2
    diagonal_collisions = 0
    n = len(chromosome)
    left_diagonal = [0] * (2 * n - 1)
    right_diagonal = [0] * (2 * n - 1)

    for i in range(n):
        left_diagonal[i + chromosome[i] - 1] += 1
        right_diagonal[len(chromosome) - i + chromosome[i] - 2] += 1

    for i in range(2 * n - 1):
        counter = 0
        if left_diagonal[i] > 1:
            counter += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            counter += right_diagonal[i] - 1
        diagonal_collisions += counter

    return int(maxFitness - (horizontal_collisions + diagonal_collisions))

def crossover(x, y):
    n = len(x)
    child = [0] * n
    for i in range(n):
        c = random.randint(0, 1)
        if c < 0.5:
            child[i] = x[i]
        else:
            child[i] = y[i]
    return child

def mutate(x):
    n = len(x)
    c = random.randint(0, n - 1)
    m = random.randint(0, n - 1)
    x[c] = m
    return x

def probability(chromosome, maxFitness):
    return fitness(chromosome, maxFitness) / maxFitness

def random_pick(population, probabilities):
    population_with_probability = zip(population, probabilities)
    total = sum(w for c, w in population_with_probability)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(population, probabilities):
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"

def genetic_queen(population, maxFitness):
    mutation_probability = 0.1
    new_population = []
    sorted_population = []
    probabilities = []

    for n in population:
        f = fitness(n, maxFitness)
        probabilities.append(f / maxFitness)
        sorted_population.append([f, n])

    sorted_population.sort(reverse=True)
    new_population.append(sorted_population[0][1])
    new_population.append(sorted_population[-1][1])

    for i in range(len(population) - 2):
        chromosome_1 = random_pick(population, probabilities)
        chromosome_2 = random_pick(population, probabilities)
        child = crossover(chromosome_1, chromosome_2)

        if random.random() < mutation_probability:
            child = mutate(child)
        new_population.append(child)
        if fitness(child, maxFitness) == maxFitness:
            break

    return new_population

def print_chromosome(chrom, maxFitness):
    print("Chromosome = {}, Fitness = {}".format(str(chrom), fitness(chrom, maxFitness)))

def print_board(chrom):
    board = []
    for x in range(nq):
        board.append(["x"] * nq)
    for i in range(nq):
        board[chrom[i]][i] = "Q"
    for row in board:
        print(" ".join(row))
    print()

if __name__ == "__main__":
    POPULATION_SIZE = 500
    while True:
        nq = int(input("Please enter your desired number of queens (0 for exit): "))
        if nq == 0:
            break
        maxFitness = (nq * (nq - 1)) / 2
        population = [random_chromosome(nq) for _ in range(POPULATION_SIZE)]
        generation = 1
        while not maxFitness in [fitness(chrom, maxFitness) for chrom in population] and generation < 200:
            population = genetic_queen(population, maxFitness)
            if generation % 10 == 0:
                print(" === Generation {} === ".format(generation))
                print("Maximum Fitness = {}".format(max([fitness(n, maxFitness) for n in population])))
            generation += 1
        fitnessOfChromosomes = [fitness(chrom, maxFitness) for chrom in population]

        bestChromosomes = population[fitnessOfChromosomes.index(max(fitnessOfChromosomes))]

        if maxFitness in fitnessOfChromosomes:
            print_chromosome(bestChromosomes, maxFitness)
            print_board(bestChromosomes)
        else:
            print(
                "Unfortunately, we couldn't find the answer until generation {}. The best answer that the algorithm found was:".format(generation - 1)
            )
            print_board(bestChromosomes)
