from autocurricula.league_trainer import LeagueTrainer
from autocurricula.league_config import LeagueConfig
from autocurricula.defaults import default_peft_config

import pytest


@pytest.fixture
def ac_trainer():
    ac_config = LeagueConfig(
        generations=4,
        rounds=2,
        matches=10,
        ma_weight=0.4,
        me_weight=0.2,
        le_weight=0.4,
    )
    ac_trainer = LeagueTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_league_entry(ac_trainer):
    # TODO: Rename generations to num_gens, num_seasons. Also current_gen
    entrants_by_gen = []
    for ac_trainer.current_gen in range(ac_trainer.ac_config.generations):
        entrants_by_gen += [ac_trainer.entry()]

    # There's just a new main agent, then also a new main exploiter, then also a new league exploiter.
    assert len(entrants_by_gen[0]) < len(entrants_by_gen[1])
    assert len(entrants_by_gen[1]) < len(entrants_by_gen[2])
    assert len(entrants_by_gen[2]) == len(entrants_by_gen[3])


def test_league_match(ac_trainer):
    matches_by_gen = []

    for ac_trainer.current_gen in range(ac_trainer.ac_config.generations):
        ac_trainer.accommodate_entrants(ac_trainer.entry())
        matches_by_gen += [ac_trainer.match()]

    assert all([m == matches_by_gen[0][0] for m in matches_by_gen[0]])
    assert len(matches_by_gen[1]) == len(matches_by_gen[2])
    assert all([isinstance(m, tuple) for m in matches_by_gen[0]])
    assert all([isinstance(p, dict) for p in matches_by_gen[0][0]])