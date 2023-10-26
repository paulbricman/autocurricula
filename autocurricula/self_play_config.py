from autocurricula.autocurriculum_config import AutocurriculumConfig


class SelfPlayConfig(AutocurriculumConfig):
    def __init__(
        self,
        generations: int,
        rounds: int,
        matches: int,
    ):
        super().__init__(generations, rounds)
        self.matches = matches
