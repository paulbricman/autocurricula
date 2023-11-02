from autocurricula.fictitious_gan_trainer import FictitiousGANTrainer
from autocurricula.defaults import default_peft_config

import pytest


@pytest.fixture
def ac_trainer():
    ac_trainer = FictitiousGANTrainer()
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_league_entry(ac_trainer):
    entrants_by_epoch = []
    for ac_trainer.current_epoch in range(ac_trainer.ac_config.epochs):
        entrants_by_epoch += [ac_trainer.entry()]

    # We get one new G and one new D each epoch.
    assert all([len(e) == 2 for e in entrants_by_epoch])


def test_league_match(ac_trainer):
    matches_by_epoch = []

    for ac_trainer.current_epoch in range(ac_trainer.ac_config.epochs):
        ac_trainer.accommodate_entrants(ac_trainer.entry())
        matches_by_epoch += [ac_trainer.match()]

    assert all([len(e) == 2 * (e_idx + 1) for e_idx, e in enumerate(matches_by_epoch)])
