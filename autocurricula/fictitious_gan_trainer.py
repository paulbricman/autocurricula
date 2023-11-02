from autocurricula import AutocurriculumTrainer
from autocurricula.fictitious_gan_config import FictitiousGANConfig


class FictitiousGANTrainer(AutocurriculumTrainer):
    def __init__(self, ac_config=FictitiousGANConfig()):
        assert isinstance(ac_config, FictitiousGANConfig)
        super().__init__(ac_config)

    def entry(self):
        # We get one new G and one new D each epoch.
        return [
            {"role": "generator"},
            {"role": "discriminator"},
        ]

    def match(self):
        gs = [e for e in self.players if e["role"] == "generator"]
        # The "epoch" field gets automatically populated at entry.
        latest_g = sorted(gs, key=lambda x: x["epoch"])[-1]

        ds = [e for e in self.players if e["role"] == "discriminator"]
        latest_d = sorted(ds, key=lambda x: x["epoch"])[-1]

        # Every round, latest players play all compatible past players.
        return [(latest_g, d) for d in ds] + [(latest_d, g) for g in gs]
