import numpy as np

class AntColonyOptimization:
    def __init__(self, distances, num_ants, num_iterations, alpha=1, beta=2, rho=0.1, Q=1, init_pheromone=1):
        self.distances = distances
        self.num_cities = len(distances)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone factor
        self.beta = beta  # Heuristic factor
        self.rho = rho  # Evaporation rate
        self.Q = Q  # Pheromone deposit factor
        self.init_pheromone = init_pheromone
        self.pheromone = np.ones((self.num_cities, self.num_cities)) * init_pheromone
        self.best_path = None
        self.best_distance = float('inf')

    def run(self):
        for iteration in range(self.num_iterations):
            ant_paths = self.generate_ant_paths()
            self.update_pheromone(ant_paths)
            best_ant_path = min(ant_paths, key=lambda x: self.get_path_distance(x))
            best_distance = self.get_path_distance(best_ant_path)
            if best_distance < self.best_distance:
                self.best_path = best_ant_path
                self.best_distance = best_distance
            print("Iteration {}: Best Distance: {}".format(iteration + 1, self.best_distance))
        return self.best_path, self.best_distance

    def generate_ant_paths(self):
        ant_paths = []
        for _ in range(self.num_ants):
            visited = [False] * self.num_cities
            current_city = 0
            path = [current_city]
            visited[current_city] = True
            while len(path) < self.num_cities:
                next_city = self.select_next_city(current_city, visited)
                path.append(next_city)
                visited[next_city] = True
                current_city = next_city
            path.append(path[0])  # Complete the loop
            ant_paths.append(path)
        return ant_paths

    def select_next_city(self, current_city, visited):
        probabilities = self.calculate_probabilities(current_city, visited)
        return np.random.choice(range(self.num_cities), p=probabilities)

    def calculate_probabilities(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        distances = self.distances[current_city]
        unvisited_cities = np.where(~np.array(visited))[0]
        heuristic = 1 / (distances + 1e-10)  # Adding a small value to avoid division by zero
        probabilities = np.zeros_like(pheromone)
        probabilities[unvisited_cities] = (pheromone[unvisited_cities] ** self.alpha) * (heuristic[unvisited_cities] ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def get_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distances[path[i], path[i + 1]]
        return distance

    def update_pheromone(self, ant_paths):
        self.pheromone *= (1 - self.rho)
        for path in ant_paths:
            path_distance = self.get_path_distance(path)
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i + 1]] += self.Q / path_distance
                self.pheromone[path[i + 1], path[i]] += self.Q / path_distance

# Example usage
if __name__ == "__main__":
    # Example distance matrix (replace with your own)
    distances = np.array([[0, 10, 15, 20],
                          [10, 0, 35, 25],
                          [15, 35, 0, 30],
                          [20, 25, 30, 0]])

    num_ants = 10
    num_iterations = 10
    alpha = 1
    beta = 2
    rho = 0.1
    Q = 1
    init_pheromone = 1
    aco = AntColonyOptimization(distances, num_ants, num_iterations, alpha, beta, rho, Q, init_pheromone)
    best_path, best_distance = aco.run()
    print("Best Path:", best_path)
    print("Best Distance:", best_distance)
