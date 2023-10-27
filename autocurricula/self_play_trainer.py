from autocurricula.autocurriculum_trainer import AutocurriculumTrainer
from autocurricula.self_play_config import SelfPlayConfig

from typing import List, Dict, Tuple
import random


class SelfPlayTrainer(AutocurriculumTrainer):
    def __init__(self, ac_config: SelfPlayConfig = SelfPlayConfig()):
        assert isinstance(ac_config, SelfPlayConfig)
        super().__init__(ac_config)

    def entry(self) -> List[Dict]:
        return [{}]

    def match(self) -> List[Tuple]:
        current_player = [p for p in self.players if p["epoch"] == self.current_epoch][
            0
        ]

        if self.current_epoch < 1:
            return [
                (current_player, current_player) for _ in range(self.ac_config.matches)
            ]

        previous_player = [
            p for p in self.players if p["epoch"] == self.current_epoch - 1
        ][0]
        return [
            (current_player, previous_player) for _ in range(self.ac_config.matches)
        ]
