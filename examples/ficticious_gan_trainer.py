from autocurricula import AutocurriculumConfig, AutocurriculumTrainer


class FictitiousGANConfig(AutocurriculumConfig):
    def __init__(self, generations: int = 4, rounds: int = 2):
        super().__init__(generations, rounds)


class FictitiousGANTrainer(AutocurriculumTrainer):
    def __init__(self, ac_config):
        assert isinstance(ac_config, FictitiousGANConfig)
        super().__init__(ac_config)

    def entry(self):
        # We get one new G and one new D each generation.
        return [
            {"role": "generator"},
            {"role": "discriminator"},
        ]

    def match(self):
        gs = [e for e in self.players if e["role"] == "generator"]
        # The "gen" field gets automatically populated for entrants.
        latest_g = sorted(gs, key=lambda x: x["gen"])[-1]

        ds = [e for e in self.players if e["role"] == "discriminator"]
        latest_d = sorted(ds, key=lambda x: x["gen"])[-1]

        # Every round, latest players play all compatible past players.
        return [(latest_g, d) for d in ds] + [(latest_d, g) for g in gs]
