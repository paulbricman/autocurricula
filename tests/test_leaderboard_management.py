from autocurricula.league_trainer import LeagueTrainer
from autocurricula.league_config import LeagueConfig
from autocurricula.defaults import default_peft_config

import pytest
import json


@pytest.fixture
def ac_trainer():
    ac_config = LeagueConfig(
        generations=2,
        rounds=2,
        matches=10,
        ma_weight=0.4,
        me_weight=0.2,
        le_weight=0.4,
    )
    ac_trainer = LeagueTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_populate_leaderboard(ac_trainer):
    entrants = [{"gen": g} for g in range(3)]
    ac_trainer.accommodate_entrants(entrants)

    assert len(ac_trainer.leaderboard) == len(entrants)


def test_update_leaderboard(ac_trainer):
    entrants = [{"gen": g} for g in range(3)]
    matches = [(entrants[0], entrants[1]), (entrants[1], entrants[2])] * 10
    evals = [[(-1, 1)], [(-1, 1)]] * 10

    ac_trainer.accommodate_entrants(entrants)
    ac_trainer.update_leaderboard(matches, evals)

    assert (
        ac_trainer.leaderboard[json.dumps(entrants[0])]
        < ac_trainer.leaderboard[json.dumps(entrants[2])]
    )
