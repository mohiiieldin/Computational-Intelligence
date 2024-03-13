from genetic_algorithm import GeneticAlgorithm
import random
import time
from typing import List
import matplotlib.pyplot as plt


class OneMaxGA(GeneticAlgorithm):
    """
    Genetic algorithm for solving the One-Max problem.
    Inherits from the GeneticAlgorithm abstract base class.
    """

    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        crossover_prob: float,
        mutation_rate: float,
        elitism_num: int,
    ):
        """
        Initialize the OneMaxGA instance.

        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Length of each chromosome (bitstring).
            mutation_rate (float): Probability of mutation for each bit.
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.elitism_num = elitism_num
        self.population = self.initialize_population()

    def create_individual(self) -> List[int]:
        """
        Create a new individual (random bitstring).

        Returns:
            List[int]: A newly created individual.
        """
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]

    def initialize_population(self) -> List[List[int]]:
        """
        Initialize the population with random bitstrings.

        Returns:
            List[List[int]]: Initial population.
        """

        return [self.create_individual() for _ in range(self.population_size)]

    def evaluate_fitness(self, chromosome: List[int]) -> int:
        """
        Evaluate the fitness of an individual (sum of 1s in the bitstring).

        Args:
            chromosome (List[int]): The bitstring representing an individual.

        Returns:
            int: Fitness value.
        """
        count_ones = sum(1 for bit in chromosome if bit == 1)
        return count_ones

    def calculate_cumulative_probabilities(self) -> List[float]:
        """
        Calculate cumulative probabilities for each individual.

        Returns:
            List[float]: Cumulative probabilities.
        """
        total_fitness = sum(
            self.evaluate_fitness(chromosome) for chromosome in self.population
        )
        relative_fitness = [
            self.evaluate_fitness(chromosome) / total_fitness
            for chromosome in self.population
        ]
        cumulative_probabilities = [relative_fitness[0]]
        for i in range(1, self.population_size):
            cumulative_probabilities.append(
                cumulative_probabilities[i - 1] + relative_fitness[i]
            )
        return cumulative_probabilities

    def select_parents(self) -> List[List[int]]:
        """
        Select parents based on cumulative probabilities.

        Returns:
            List[List[int]]: Selected parents.
        """
        cumulative_probabilities = self.calculate_cumulative_probabilities()
        selected_parents = random.choices(
            self.population, cum_weights=cumulative_probabilities, k=2
        )
        return selected_parents

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[List[int]]:
        """
        Perform one-point crossover between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """

        if random.uniform(0, 1) < self.crossover_prob:
            # TODO
            crossover_point = random.randint(1, self.chromosome_length - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            return offspring1, offspring2
        else:
            return parent1, parent2

    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Apply bit flip mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.uniform(0, 1) < self.mutation_rate:
                # TODO  # Bit flip
                chromosome[i] = 1 - chromosome[i]
        return mutated_chromosome

    def elitism(self) -> List[List[int]]:
        """
        Apply elitism to the population (keep the best two individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        sorted_population = sorted(
            self.population, key=self.evaluate_fitness, reverse=True
        )
        # TODO #return the best elitism_num
        return sorted_population[: self.elitism_num]

    def run(self, max_generations):
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            new_population = new_population[
                0 : self.population_size - self.elitism_num
            ]  # make sure the new_population is the same size of original population - the best individuals we will append next
            best_individuals = self.elitism()
            new_population.extend(best_individuals)
            self.population = new_population
        plt.hist(
            [self.evaluate_fitness(individual) for individual in self.population],
            bins=10,
        )
        best_solution = max(self.population, key=self.evaluate_fitness)
        plt.show()
        return best_solution


if __name__ == "__main__":
    population_size = 5
    chromosome_length = 4
    crossover_prob = 0.6
    mutation_rate = 0.05
    elitism_num = 2
    max_generations = 100
    start = time.time()
    onemax_ga = OneMaxGA(
        population_size, chromosome_length, crossover_prob, mutation_rate, elitism_num
    )

    best_fitness_values = []  # Store best fitness values for each generation
    average_fitness_values = []  # Store average fitness values for each generation
    for generation in range(max_generations):
        new_population = []
        fitness = []
        while len(new_population) < onemax_ga.population_size:
            parent1, parent2 = onemax_ga.select_parents()
            offspring1, offspring2 = onemax_ga.crossover(parent1, parent2)
            offspring1 = onemax_ga.mutate(offspring1)
            offspring2 = onemax_ga.mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        new_population = new_population[
            0 : onemax_ga.population_size - elitism_num
        ]  # make sure the new_population is the same size of original population - the best individuals we will append next
        best_individuals = onemax_ga.elitism()
        new_population.extend(best_individuals)
        onemax_ga.population = new_population

        # Calculate best fitness value for the current generation
        best_fitness = max(
            onemax_ga.evaluate_fitness(individual)
            for individual in onemax_ga.population
        )
        best_fitness_values.append(best_fitness)

        # Calculate average fitness value for the current generation
        average_fitness = sum(
            onemax_ga.evaluate_fitness(individual)
            for individual in onemax_ga.population
        ) / len(onemax_ga.population)
        average_fitness_values.append(average_fitness)

    # Retrieve and print the best solution after running the genetic algorithm
    best_solution = max(onemax_ga.population, key=onemax_ga.evaluate_fitness)
    print(f"Best solution: {best_solution}")
    print(f"Fitness: {onemax_ga.evaluate_fitness(best_solution)}")
    # for individual in best_fitness_values[:10]:
    #     print(individual)
    # for individual in average_fitness_values[:10]:
    #     print(individual)
    print(len(best_fitness_values))
    print(len(average_fitness_values))
    # Plotting best and average fitness values over generations
    plt.plot(range(max_generations), best_fitness_values, label="Best Fitness")
    plt.plot(range(max_generations), average_fitness_values, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best and Average Fitness Values Over Generations")
    plt.legend()
    plt.show()

    ga_time = time.time() - start
    print("GA Solution Time:", round(ga_time, 1), "Seconds")
