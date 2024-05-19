import random
import math

def generate_initial_population(num_cities, population_size):
    population = []
    for _ in range(population_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

# Function to calculate total distance of a route using distances matrix
def total_distance(route, distances):
    total = 0
    for i in range(len(route)):
        total += distances[route[i - 1]][route[i]]
    return total

# Tournament selection
def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(range(len(population)), tournament_size)
        winner = min(participants, key=lambda x: fitness[x])
        selected.append(population[winner])
    return selected

# Partially mapped crossover
def partially_mapped_crossover(parent1, parent2):
    n = len(parent1)
    start = random.randint(0, n - 1)
    end = random.randint(start + 1, n)
    child = [-1] * n
    for i in range(start, end):
        child[i] = parent1[i]
    for i in range(n):
        if parent2[i] not in child[start:end]:
            idx = parent2.index(parent1[i])
            while child[idx] != -1:
                idx = parent2.index(parent1[idx])
            child[idx] = parent2[i]
    return child

# Scramble mutation
def scramble_mutation(chromosome):
    start = random.randint(0, len(chromosome) - 1)
    end = random.randint(start + 1, len(chromosome))
    segment = chromosome[start:end]
    random.shuffle(segment)
    chromosome[start:end] = segment
    return chromosome

# Genetic algorithm
def genetic_algorithm(population_size, generations, distances, tournament_size, mutation_rate):
    num_cities = len(distances)
    population = generate_initial_population(num_cities, population_size)
    best_individual = None
    best_fitness = float('-inf')
    convergence_count = 0
    
    for generation in range(generations):
        fitness = [1 / total_distance(route, distances) for route in population]
        max_fitness = max(fitness)
        
        print(f"=== Generation {generation + 1} ===")
        print(f"Maximum Fitness = {max_fitness:.2f}")
        
        if max_fitness > best_fitness:
            best_index = fitness.index(max_fitness)
            best_individual = population[best_index]
            best_fitness = max_fitness
            convergence_count = 0
        else:
            convergence_count += 1
        
        if convergence_count >= 50:
            print("Convergence reached. Stopping the algorithm.")
            break
        
        selected_parents = tournament_selection(population, fitness, tournament_size)
        offspring = []
        
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1 = partially_mapped_crossover(parent1, parent2)
            child2 = partially_mapped_crossover(parent2, parent1)
            offspring.extend([child1, child2])
        
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = scramble_mutation(offspring[i])
        
        population = sorted(population, key=lambda x: total_distance(x, distances))[:len(population) - len(offspring)]
        population.extend(offspring)
    
    shortest_distance = total_distance(best_individual, distances)
    return best_individual, shortest_distance

# Example usage
distances_matrix = [
    [0, 10, 15, 20, 5],
    [10, 0, 35, 25, 15],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 10],
    [5, 15, 20, 10, 0]
]

population_size = 100
generations = 1000
tournament_size = 5
mutation_rate = 0.1

best_route, shortest_distance = genetic_algorithm(population_size, generations, distances_matrix, tournament_size, mutation_rate)

print("\nBest route:", best_route)
print("Shortest distance:", shortest_distance)
