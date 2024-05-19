# """Solve the Optimization problem using Particle Swarm Optimization
# Maxization f(a1, a2, a3, a4, a5) = 1 + 2a1 + (3a2 - 1) + 3a3 + 2a4 square + (5a5 + 2)
# where, n = 20, Wmax = 0.9, Wmin = 0.3"""

# import random
# import numpy as np

# # Define the function to be maximized
# def fitness_function(a):
#     a1, a2, a3, a4, a5 = a
#     return 1 + 2 * a1 + (3 * a2 - 1) + 3 * a3 + 2 * (a4 ** 2) + (5 * a5 + 2)

# # PSO parameters
# n_particles = 20
# Wmax = 0.9
# Wmin = 0.3
# c1 = 2
# c2 = 2
# n_iterations = 100

# # Bounds for each variable
# bounds = [(10, 60), (15, 30), (25, 75), (10, 30), (10, 50)]

# # Initialize the particles
# particles = []
# velocities = []
# for _ in range(n_particles):
#     particle = [random.uniform(bound[0], bound[1]) for bound in bounds]
#     particles.append(particle)
#     velocities.append([0] * len(bounds))

# # Initialize the best positions
# pbest = particles[:]
# pbest_fitness = [fitness_function(p) for p in particles]
# gbest = max(particles, key=fitness_function)
# gbest_fitness = fitness_function(gbest)

# # PSO main loop
# for t in range(n_iterations):
#     w = Wmax - (Wmax - Wmin) * (t / n_iterations)
#     for i in range(n_particles):
#         r1 = random.random()
#         r2 = random.random()
#         velocities[i] = [
#             w * velocities[i][j] + c1 * r1 * (pbest[i][j] - particles[i][j]) + c2 * r2 * (gbest[j] - particles[i][j])
#             for j in range(len(bounds))
#         ]
#         particles[i] = [
#             particles[i][j] + velocities[i][j]
#             for j in range(len(bounds))
#         ]
#         particles[i] = [
#             np.clip(particles[i][j], bounds[j][0], bounds[j][1])
#             for j in range(len(bounds))
#         ]
        
#         current_fitness = fitness_function(particles[i])
#         if current_fitness > pbest_fitness[i]:
#             pbest[i] = particles[i]
#             pbest_fitness[i] = current_fitness
        
#         if current_fitness > gbest_fitness:
#             gbest = particles[i]
#             gbest_fitness = current_fitness

#     print(f"Iteration {t+1}/{n_iterations}, Best Fitness: {gbest_fitness}")

# print("\nOptimal Solution:")
# print("Variables:", gbest)
# print("Maximum Fitness:", gbest_fitness)


import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([0.0 for _ in bounds])
        self.best_position = np.copy(self.position)
        self.best_value = float('-inf')

def objective_function(a):
    a1, a2, a3, a4, a5 = a
    return 1 + 2*a1 + (3*a2 - 1) + 3*a3 + 2*(a4**2) + (5*a5 + 2)

def particle_swarm_optimization(bounds, num_particles, max_iter, w_max, w_min):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_value = float('-inf')
    global_best_position = None
    
    for t in range(max_iter):
        print(f"\nIteration {t+1}")
        w = w_max - (w_max - w_min) * t / max_iter
        
        for particle in particles:
            fitness_value = objective_function(particle.position)
            
            if fitness_value > particle.best_value:
                particle.best_value = fitness_value
                particle.best_position = np.copy(particle.position)
                
            if fitness_value > global_best_value:
                global_best_value = fitness_value
                global_best_position = np.copy(particle.position)
        
        for i, particle in enumerate(particles):
            print(f"\nParticle {i+1}")
            print(f"Position: {particle.position}")
            print(f"Velocity: {particle.velocity}")
            print(f"Personal Best Position: {particle.best_position}")
            print(f"Personal Best Value: {particle.best_value}")

            cognitive_component = np.random.uniform(0, 1, len(bounds)) * (particle.best_position - particle.position)
            social_component = np.random.uniform(0, 1, len(bounds)) * (global_best_position - particle.position)
            new_velocity = w * particle.velocity + cognitive_component + social_component
            new_position = particle.position + new_velocity
            
            # Clip positions to be within bounds
            new_position = np.clip(new_position, [low for low, high in bounds], [high for low, high in bounds])
            
            print(f"Updated Velocity: {new_velocity}")
            print(f"Updated Position: {new_position}")

            particle.velocity = new_velocity
            particle.position = new_position
    
    print("\nBest Solution Found")
    print(f"Best Position: {global_best_position}")
    print(f"Best Value: {global_best_value}")
    return global_best_position, global_best_value

# Problem bounds
bounds = [(10, 60), (15, 30), (25, 75), (10, 30), (10, 50)]

# Parameters for PSO
num_particles = 5  # Reduced number for more manageable output
max_iter = 10     # Reduced number of iterations for demonstration
w_max = 0.9
w_min = 0.3

# Run PSO
best_position, best_value = particle_swarm_optimization(bounds, num_particles, max_iter, w_max, w_min)
