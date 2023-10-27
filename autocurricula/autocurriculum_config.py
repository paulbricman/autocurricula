from abc import ABC


class AutocurriculumConfig(ABC):
    def __init__(self, epochs: int = 4, rounds: int = 2):
        self.epochs = epochs
        self.rounds = rounds
