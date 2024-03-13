from genetic_algorithm import GeneticAlgorithm
import random
import time
import statistics
from typing import List

class OneMaxGA(GeneticAlgorithm):
    """
    Genetic algorithm for solving the One-Max problem.
    Inherits from the GeneticAlgorithm abstract base class.
    """

    def __init__(self, population_size: int, chromosome_length: int, crossover_prob:float, mutation_rate: float, elitism_num: int):
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
        return sum(chromosome)
    
    def calculate_cumulative_probabilities(self) -> List[float]:
        """
        Calculate cumulative probabilities for each individual.

        Returns:
            List[float]: Cumulative probabilities.
        """
        population= self.initialize_population()
        fitness_values = [self.evaluate_fitness(individual) for individual in population]
        fitness_sum = sum(fitness_values)
        probabilities = [f / fitness_sum for f in fitness_values]

        cumulative = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]
        #probabilities[:i + 1] makes a sublist from begain to element at index i
        return cumulative


    def select_parents(self) -> List[List[int]]:
        """
        Select parents based on cumulative probabilities.

        Returns:
            List[List[int]]: Selected parents.
        """
        cumulative_probabilities = self.calculate_cumulative_probabilities()
        selected_parents = random.choices(self.population, cum_weights = cumulative_probabilities, k = 2)
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
            split_point= round(random.uniform(0, 1)*len(parent1))
            return parent1[:split_point] + parent2[split_point:], parent2[:split_point] + parent1[split_point:]
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
                mutated_chromosome[i] = 1 if mutated_chromosome[i] == 0 else 0      # Bit flip
        return mutated_chromosome

    def elitism(self) -> List[List[int]]:
        """
        Apply elitism to the population (keep the best two individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        sorted_population = sorted(self.population, key=self.evaluate_fitness, reverse=True)
        return sorted_population[0],sorted_population[1]       #return the best elitism_num 


    def run(self, max_generations):
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            new_population = new_population[0:self.population_size-self.elitism_num] # make sure the new_population is the same size of original population - the best individuals we will append next
            best_individuals = self.elitism()
            new_population.extend(best_individuals)
            self.population = new_population


        best_solution = max(self.population, key=self.evaluate_fitness)
        return best_solution

if __name__ == "__main__":
    population_size = 20
    chromosome_length = 5
    crossover_prob = 0.6
    mutation_rate = 0.05
    elitism_num = 2
    max_generations = 100
    run=10

    bestSolutionFitness_list = []
    ga_time_list = []

    for _ in range (10):
        start = time.time()
        onemax_ga = OneMaxGA(population_size, chromosome_length,crossover_prob, mutation_rate,elitism_num)
        best_solution = onemax_ga.run(max_generations)
        ga_time = time.time()-start
        bestSolutionFitness = onemax_ga.evaluate_fitness(best_solution)
        print("GA Solution Time:",round(ga_time,1),'Seconds')
        print(f"Best solution: {best_solution}")
        print(f"Fitness: {bestSolutionFitness}")

        bestSolutionFitness_list.append(bestSolutionFitness)
        ga_time_list.append(ga_time)



    # Analyzing bestSolutionFitness_list
    mean_bestSolutionFitness = statistics.mean(bestSolutionFitness_list)
    median_bestSolutionFitness = statistics.median(bestSolutionFitness_list)
    min_bestSolutionFitness = min(bestSolutionFitness_list)
    max_bestSolutionFitness = max(bestSolutionFitness_list)
    stddev_bestSolutionFitness = statistics.stdev(bestSolutionFitness_list)

    # Analyzing ga_time_list
    mean_ga_time = statistics.mean(ga_time_list)
    median_ga_time = statistics.median(ga_time_list)
    min_ga_time = min(ga_time_list)
    max_ga_time = max(ga_time_list)
    stddev_ga_time = statistics.stdev(ga_time_list)

    # Print the statistics
    print("Statistics for Best Solution Fitness list:")
    print(f"Mean: {mean_bestSolutionFitness}")
    print(f"Median: {median_bestSolutionFitness}")
    print(f"Min: {min_bestSolutionFitness}")
    print(f"Max: {max_bestSolutionFitness}")
    print(f"Standard Deviation: {stddev_bestSolutionFitness}")


    print("\nStatistics for time list:")
    print(f"Mean: {mean_ga_time}")
    print(f"Median: {median_ga_time}")
    print(f"Min: {min_ga_time}")
    print(f"Max: {max_ga_time}")
    print(f"Standard Deviation: {stddev_ga_time}")
