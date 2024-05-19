import random

class NQueensGeneticAlgorithm:
    def __init__(self, n, population_size=100, mutation_rate=0.01):
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = list(range(1, self.n + 1))
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        clashes = 0
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if abs(i - j) == abs(chromosome[i] - chromosome[j]):
                    clashes += 1
        return 1 / (clashes + 1)

    def select_parent(self):
        return random.choice(self.population)

    def crossover_multipoint(self, parent1, parent2):
        crossover_points = sorted(random.sample(range(1, self.n), 2))
        child = [-1] * self.n
        for i in range(crossover_points[0], crossover_points[1]):
            child[i] = parent1[i]
        index = 0
        for i in range(self.n):
            if child[i] == -1:
                while parent2[index] in child:
                    index += 1
                child[i] = parent2[index]
                index += 1
        return child

    def crossover_uniform(self, parent1, parent2):
        child = [-1] * self.n
        for i in range(self.n):
            if random.random() < 0.5:
                child[i] = parent1[i]
        for i in range(self.n):
            if child[i] == -1:
                for j in range(self.n):
                    if parent2[j] not in child:
                        child[i] = parent2[j]
                        break
        return child

    def crossover_three_parent(self, parent1, parent2, parent3):
        child = [-1] * self.n
        for i in range(self.n):
            if i % 3 == 0:
                child[i] = parent1[i]
            elif i % 3 == 1:
                child[i] = parent2[i]
            else:
                child[i] = parent3[i]
        return child

    def crossover_shuffle(self, parent1, parent2):
        crossover_point = random.randint(0, self.n - 1)
        child = parent1[:crossover_point]
        for gene in parent2:
            if gene not in child:
                child.append(gene)
        return child

    def mutate(self, child):
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(self.n), 2)
            child[index1], child[index2] = child[index2], child[index1]

    def evolve(self):
        next_population = []
        for _ in range(self.population_size):
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover_multipoint(parent1, parent2) # Change crossover method here
            self.mutate(child)
            next_population.append(child)
        self.population = next_population

    def get_best_solution(self):
        best_chromosome = max(self.population, key=self.fitness)
        return best_chromosome, self.fitness(best_chromosome)

# Example usage
if __name__ == "__main__":
    n = 8  # Size of the board
    generations = 1000
    ga = NQueensGeneticAlgorithm(n)
    for _ in range(generations):
        ga.evolve()
    best_solution, fitness = ga.get_best_solution()
    print("Best solution:", best_solution)
    print("Fitness:", fitness)
