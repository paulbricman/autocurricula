from abc import ABC


class AutocurriculumConfig(ABC):
    def __init__(self, generations: int = 4, rounds: int = 2):
        self.generations = generations
        self.rounds = rounds
