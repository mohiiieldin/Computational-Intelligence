from abc import ABC, abstractmethod


class GeneticAlgorithm(ABC):
    """
    Abstract base class for genetic algorithms.
    """

    @abstractmethod
    def initialize_population(self):
        """
        Initialize the population of individuals.

        Returns:
            List[List[int]]: A list of bitstrings (chromosomes).
        """
        pass

    @abstractmethod
    def evaluate_fitness(self, chromosome):
        """
        Evaluate the fitness of an individual.

        Args:
            chromosome (List[int]): The bitstring representing an individual.

        Returns:
            int: Fitness value (e.g., sum of 1s in the bitstring).
        """
        pass

    @abstractmethod
    def select_parents(self):
        """
        Select two parent chromosomes from the population.

        Returns:
            List[List[int]]: Two parent chromosomes.
        """
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        """
        Perform crossover (recombination) between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """
        pass

    @abstractmethod
    def mutate(self, chromosome):
        """
        Apply mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        pass

    @abstractmethod
    def elitism(self, new_population):
        """
        Apply elitism to the population (e.g., keep the best individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        pass
