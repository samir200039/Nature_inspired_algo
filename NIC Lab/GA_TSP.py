import random
import math

def generate_initial_population(num_cities, population_size):
    population = []
    for _ in range(population_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

def calculate_fitness(individual, distance_matrix):
    total_distance = 0
    num_cities = len(individual)
    for i in range(num_cities):
        current_city = individual[i]
        next_city = individual[(i + 1) % num_cities]
        total_distance += distance_matrix[current_city][next_city]
    return 1 / total_distance

def rank_based_selection(population, fitness_values):
    ranked_individuals = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    ranks = [i + 1 for i in range(len(ranked_individuals))]
    rank_sum = sum(ranks)
    probabilities = [rank / rank_sum for rank in ranks]
    selected_index = random.choices(range(len(population)), probabilities)[0]
    return ranked_individuals[selected_index][0]

def cycle_crossover(parent1, parent2):
    num_cities = len(parent1)
    child = [-1] * num_cities
    cycle_start = 0
    while cycle_start < num_cities:
        child[cycle_start] = parent1[cycle_start]
        next_city = parent2[cycle_start]
        while next_city != parent1[cycle_start]:
            index = parent1.index(next_city)
            child[index] = parent1[index]
            next_city = parent2[index]
        cycle_start += 1
        while cycle_start < num_cities and child[cycle_start] != -1:
            cycle_start += 1
    for i in range(num_cities):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def inversion_mutation(individual):
    num_cities = len(individual)
    start, end = random.sample(range(num_cities), 2)
    if start > end:
        start, end = end, start
    individual[start:end+1] = reversed(individual[start:end+1])
    return individual

def genetic_algorithm(distance_matrix, population_size, num_generations, mutation_rate):
    num_cities = len(distance_matrix)
    population = generate_initial_population(num_cities, population_size)
    best_individual = None
    best_fitness = float('-inf')
    iterations = 0
    convergence_count = 0

    for generation in range(num_generations):
        print(f"=== Generation {generation + 1} ===")
        fitness_values = [calculate_fitness(individual, distance_matrix) for individual in population]
        max_fitness = max(fitness_values)
        print(f"Maximum Fitness = {1 / max_fitness:.2f}")

        if max_fitness > best_fitness:
            best_index = fitness_values.index(max_fitness)
            best_individual = population[best_index]
            best_fitness = max_fitness
            convergence_count = 0
        else:
            convergence_count += 1

        new_population = []
        for _ in range(population_size // 2):
            parent1 = rank_based_selection(population, fitness_values)
            parent2 = rank_based_selection(population, fitness_values)
            child1 = cycle_crossover(parent1, parent2)
            child2 = cycle_crossover(parent2, parent1)
            if random.random() < mutation_rate:
                child1 = inversion_mutation(child1)
            if random.random() < mutation_rate:
                child2 = inversion_mutation(child2)
            new_population.extend([child1, child2])
        
        population = new_population
        iterations += 1

        if convergence_count >= 50:
            print("Convergence reached. Stopping the algorithm.")
            break

    print("\nBest Solution:")
    print("Route:", best_individual)
    print("Distance:", 1 / best_fitness)
    print("Number of iterations:", iterations)

distance_matrix = [
    [0, 10, 15, 20, 5],
    [10, 0, 35, 25, 15],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 10],
    [5, 15, 20, 10, 0]
]

population_size = 100
num_generations = 1000
mutation_rate = 0.1
genetic_algorithm(distance_matrix, population_size, num_generations, mutation_rate)
