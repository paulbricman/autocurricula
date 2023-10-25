from abc import ABC


class AutocurriculumConfig(ABC):
    def __init__(self, generations: int, rounds: int):
        self.generations = generations
        self.rounds = rounds
