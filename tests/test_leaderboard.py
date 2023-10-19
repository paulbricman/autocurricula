from pelt.leaderboard import new_elos
from pelt.defaults import default_config
from pelt.leaderboard import populate_leaderboard, update_leaderboard
import pytest
import json


@pytest.fixture
def config():
    return default_config()


def test_new_elos():
    olds = 1000, 1000
    news = new_elos(*olds)
    assert news[0] > olds[0] and news[1] < olds[1]


def test_populate_leaderboard(config):
    entrants = [{"generation": g} for g in range(3)]
    leaderboard = {}

    leaderboard = populate_leaderboard(leaderboard, entrants, config)
    assert len(leaderboard) == len(entrants)


def test_update_leaderboard(config):
    league = [{"generation": g} for g in range(3)]
    matches = [(league[0], league[1]), (league[1], league[2])] * 10
    evals = [[(-1, 1)], [(-1, 1)]] * 10

    leaderboard = {}
    leaderboard = populate_leaderboard(leaderboard, league, config)
    leaderboard = update_leaderboard(leaderboard, matches, evals, config)

    assert leaderboard[json.dumps(league[0])] < leaderboard[json.dumps(league[2])]
