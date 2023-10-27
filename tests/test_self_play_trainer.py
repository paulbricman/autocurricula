from autocurricula.self_play_trainer import SelfPlayTrainer
from autocurricula.self_play_config import SelfPlayConfig
from autocurricula.defaults import default_peft_config

import pytest


@pytest.fixture
def ac_trainer():
    ac_config = SelfPlayConfig(
        epochs=4,
        rounds=2,
        matches=10,
    )
    ac_trainer = SelfPlayTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_self_play_entry(ac_trainer):
    entrants_by_epoch = []
    for ac_trainer.current_epoch in range(ac_trainer.ac_config.epochs):
        entrants_by_epoch += [ac_trainer.entry()]

    # There's always just one new entrant in self-play.
    assert all([len(e) == 1 for e in entrants_by_epoch])


def test_self_play_match(ac_trainer):
    matches_by_epoch = []

    for ac_trainer.current_epoch in range(ac_trainer.ac_config.epochs):
        ac_trainer.accommodate_entrants(ac_trainer.entry())
        matches_by_epoch += [ac_trainer.match()]

    print(matches_by_epoch)

    assert all([len(e) == ac_trainer.ac_config.matches for e in matches_by_epoch])
    assert matches_by_epoch[0][0][0] == matches_by_epoch[0][0][1]
    assert set(
        [matches_by_epoch[1][0][0]["epoch"], matches_by_epoch[1][0][1]["epoch"]]
    ) == set([0, 1])
