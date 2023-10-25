from autocurricula.autocurriculum_config import AutocurriculumConfig


class LeagueConfig(AutocurriculumConfig):
    def __init__(
        self,
        generations: int,
        rounds: int,
        matches: int,
        ma_weight: float,
        me_weight: float,
        le_weight: float,
    ):
        super().__init__(generations, rounds)
        self.matches = matches
        self.ma_weight = ma_weight
        self.me_weight = me_weight
        self.le_weight = le_weight
