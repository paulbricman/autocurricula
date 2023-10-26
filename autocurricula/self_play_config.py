from autocurricula.autocurriculum_config import AutocurriculumConfig


class SelfPlayConfig(AutocurriculumConfig):
    def __init__(
        self,
        generations: int = 4,
        rounds: int = 10,
        matches: int = 100,
    ):
        super().__init__(generations, rounds)
        self.matches = matches
