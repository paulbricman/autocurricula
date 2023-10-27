from autocurricula.league_trainer import LeagueTrainer
from autocurricula.league_config import LeagueConfig
from autocurricula.defaults import default_peft_config, default_ppo_config
from autocurricula.games.tictactoe import play

import pytest
import json


@pytest.fixture
def ac_trainer():
    ac_config = LeagueConfig(
        epochs=4,
        rounds=2,
        matches=2,
        ma_weight=0.4,
        me_weight=0.2,
        le_weight=0.4,
    )
    ac_trainer = LeagueTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


@pytest.fixture
def matches():
    return [
        ({"epoch": 0}, {"epoch": 1}),
        ({"epoch": 1}, {"epoch": 0}),
        ({"epoch": 1}, {"epoch": 1}),
    ]


@pytest.fixture
def evals():
    return [[(1, -1)], [(1, -1)], [(-1, 0)]]


@pytest.fixture
def histories():
    timeline = [
        {
            "thoughts": [
                {
                    "context": "Here's the board...",
                    "behavior": "A good opening would be...",
                },
                {
                    "context": "A good opening would be...",
                    "behavior": "4",
                },
            ]
        },
        {
            "thoughts": [
                {
                    "context": "Here's the board...",
                    "behavior": "A good response would be...",
                },
                {
                    "context": "A good response would be...",
                    "behavior": "2",
                },
            ]
        },
    ]
    return [[timeline], [timeline], [timeline[:1]]]


def test_adapters(ac_trainer):
    ac_trainer.model.pretrained_model.load_adapter("adapter_params", adapter_name="A1")
    ac_trainer.model.pretrained_model.set_adapter(adapter_name="A1")

    assert list(ac_trainer.model.pretrained_model.peft_config.keys()) == [
        "default",
        "A1",
    ]
    assert ac_trainer.model.pretrained_model.active_adapter == "A1"

    ac_trainer.model.pretrained_model.load_adapter("adapter_params", adapter_name="A2")
    ac_trainer.model.pretrained_model.set_adapter(adapter_name="A2")

    assert list(ac_trainer.model.pretrained_model.peft_config.keys()) == [
        "default",
        "A1",
        "A2",
    ]
    assert ac_trainer.model.pretrained_model.active_adapter == "A2"

    ac_trainer.model.pretrained_model.set_adapter(adapter_name="default")
    assert ac_trainer.model.pretrained_model.active_adapter == "default"


def test_populate_with_entrant_adapters(ac_trainer):
    for ac_trainer.current_epoch in range(ac_trainer.ac_config.epochs):
        ac_trainer.accommodate_entrants(ac_trainer.entry())

        # There's also the default adapter.
        assert (
            len(ac_trainer.players)
            == len(ac_trainer.model.pretrained_model.peft_config) - 1
        )

    # The adapter names should actually correspond to the player dicts.
    assert [
        json.loads(e)
        for e in ac_trainer.model.pretrained_model.peft_config.keys()
        if e != "default"
    ] == ac_trainer.players


def test_trajectories_by_player(ac_trainer, matches, evals, histories):
    ac_trainer.accommodate_entrants([{}])
    ac_trainer.current_epoch += 1
    ac_trainer.accommodate_entrants([{}])

    sars = ac_trainer.trajectories_by_player(matches, evals, histories)

    assert list(sars[json.dumps({"epoch": 0})][0].values()) == [
        "Here's the board...",
        "A good opening would be...",
        1,
    ]


def test_update_players(ac_trainer, matches, evals, histories):
    ac_trainer.accommodate_entrants([{}])
    ac_trainer.current_epoch += 1
    ac_trainer.accommodate_entrants([{}])

    ac_trainer.ppo_config = default_ppo_config()
    ac_trainer.update_players(matches, evals, histories)


def test_train(ac_trainer):
    ac_trainer.train("facebook/opt-125m", play)
