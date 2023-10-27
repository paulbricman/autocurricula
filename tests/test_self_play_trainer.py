from autocurricula.self_play_trainer import SelfPlayTrainer
from autocurricula.self_play_config import SelfPlayConfig
from autocurricula.defaults import default_peft_config

import pytest


@pytest.fixture
def ac_trainer():
    ac_config = SelfPlayConfig(
        generations=4,
        rounds=2,
        matches=10,
    )
    ac_trainer = SelfPlayTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_self_play_entry(ac_trainer):
    entrants_by_gen = []
    for ac_trainer.current_gen in range(ac_trainer.ac_config.generations):
        entrants_by_gen += [ac_trainer.entry()]

    # There's always just one new entrant in self-play.
    assert all([len(e) == 1 for e in entrants_by_gen])


def test_self_play_match(ac_trainer):
    matches_by_gen = []

    for ac_trainer.current_gen in range(ac_trainer.ac_config.generations):
        ac_trainer.accommodate_entrants(ac_trainer.entry())
        matches_by_gen += [ac_trainer.match()]

    print(matches_by_gen)

    assert all([len(e) == ac_trainer.ac_config.matches for e in matches_by_gen])
    assert matches_by_gen[0][0][0] == matches_by_gen[0][0][1]
    assert set([matches_by_gen[1][0][0]["gen"], matches_by_gen[1][0][1]["gen"]]) == set(
        [0, 1]
    )
