from autocurricula.autocurriculum_trainer import AutocurriculumTrainer
from autocurricula.self_play_config import SelfPlayConfig

from typing import List, Dict, Tuple
import random


class SelfPlayTrainer(AutocurriculumTrainer):
    def __init__(self, ac_config: SelfPlayConfig):
        assert isinstance(ac_config, SelfPlayConfig)
        super().__init__(ac_config)

    def entry(self) -> List[Dict]:
        return [{}]

    def match(self) -> List[Tuple]:
        current_player = [p for p in self.players if p["gen"] == self.current_gen][0]

        if self.current_gen < 1:
            return [
                (current_player, current_player) for _ in range(self.ac_config.matches)
            ]

        previous_player = [p for p in self.players if p["gen"] == self.current_gen - 1][
            0
        ]
        return [
            (current_player, previous_player) for _ in range(self.ac_config.matches)
        ]
