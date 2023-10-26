from autocurricula.autocurriculum_config import AutocurriculumConfig


class LeagueConfig(AutocurriculumConfig):
    def __init__(
        self,
        generations: int = 4,
        rounds: int = 10,
        matches: int = 100,
        ma_weight: float = 0.4,
        me_weight: float = 0.2,
        le_weight: float = 0.4,
    ):
        super().__init__(generations, rounds)
        self.matches = matches
        self.ma_weight = ma_weight
        self.me_weight = me_weight
        self.le_weight = le_weight
