from pelt.operators import league_entry, league_match
from pelt.training import get_model_tok
from pelt.defaults import default_config
import pytest


@pytest.fixture
def model_tok(config):
    return get_model_tok("facebook/opt-125m", config)


@pytest.fixture
def config():
    return default_config()


def test_league_entry(config):
    entrants_by_generation = [
        league_entry(generation, config) for generation in range(4)
    ]

    # There's just a new main agent, then also a new main exploiter, then also a new league exploiter.
    assert len(entrants_by_generation[0]) < len(entrants_by_generation[1])
    assert len(entrants_by_generation[1]) < len(entrants_by_generation[2])
    assert len(entrants_by_generation[2]) == len(entrants_by_generation[3])


def test_league_match(config):
    league = []
    matches_by_generation = []

    for generation in range(3):
        league += league_entry(generation, config)
        matches_by_generation += [league_match(league, generation, config)]

    assert all([m == matches_by_generation[0][0] for m in matches_by_generation[0]])
    assert len(matches_by_generation[1]) == len(matches_by_generation[2])
    assert all([isinstance(m, tuple) for m in matches_by_generation[0]])
    assert all([isinstance(p, dict) for p in matches_by_generation[0][0]])
