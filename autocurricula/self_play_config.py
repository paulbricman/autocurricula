from autocurricula.autocurriculum_config import AutocurriculumConfig


class SelfPlayConfig(AutocurriculumConfig):
    def __init__(
        self,
        epochs: int = 4,
        rounds: int = 10,
        matches: int = 100,
    ):
        super().__init__(epochs, rounds)
        self.matches = matches
