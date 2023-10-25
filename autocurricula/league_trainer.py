from autocurricula.autocurriculum_trainer import AutocurriculumTrainer
from autocurricula.league_config import LeagueConfig

import random


class LeagueTrainer(AutocurriculumTrainer):
    def __init__(self, ac_config: LeagueConfig):
        assert isinstance(ac_config, LeagueConfig)
        super().__init__(ac_config)

    def entry(self):
        """
        Self-explanatory league training entry logic.
        """
        entrants = [
            {
                "role": "main_agent",
                "gen": self.current_gen,
            }
        ]
        if self.current_gen >= 1:
            entrants += [
                {
                    "role": "main_exploiter",
                    "gen": self.current_gen,
                }
            ]
        if self.current_gen >= 2:
            entrants += [
                {
                    "role": "league_exploiter",
                    "gen": self.current_gen,
                }
            ]
        return entrants

    def match(self):
        """
        League training matchmaking logic.
        """
        main_agents = [p for p in self.players if p["role"] == "main_agent"]
        current_ma = [p for p in main_agents if p["gen"] == self.current_gen][0]

        # Latest main agent plays themselves and all past versions.
        ma_matches = [(current_ma, main_agent) for main_agent in main_agents]
        legal_match_types = [ma_matches]

        if self.current_gen >= 1:
            main_exploiters = [p for p in self.players if p["role"] == "main_exploiter"]
            current_me = [p for p in main_exploiters if p["gen"] == self.current_gen][0]

            # Latest main exploiter plays latest main agent.
            me_matches = [(current_me, current_ma)]
            legal_match_types += [me_matches]

        if self.current_gen >= 2:
            league_exploiters = [
                p for p in self.players if p["role"] == "league_exploiter"
            ]
            current_le = [p for p in league_exploiters if p["gen"] == self.current_gen]

            # Latest league exploiter plays the entire league except itself.
            le_matches = [(current_le, p) for p in self.players if p is not current_le]
            legal_match_types += [le_matches]

        matches = []
        for _ in range(self.ac_config.matches):
            type = random.choices(
                range(len(legal_match_types)),
                weights=[
                    self.ac_config.ma_weight,
                    self.ac_config.me_weight,
                    self.ac_config.le_weight,
                ][: len(legal_match_types)],
            )[0]

            matches += random.choices(legal_match_types[type])

        return matches
