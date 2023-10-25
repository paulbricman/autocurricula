from autocurricula.autocurriculum_trainer import (
    get_model_tok,
    populate_with_entrant_adapters,
    train,
    trajectories_by_model,
    update,
)
from autocurricula.defaults import default_config
from autocurricula.operators import league_entry
from autocurricula.games.tictactoe import play

import pytest
import json


@pytest.fixture
def config():
    return default_config()


@pytest.fixture
def league():
    return [{"gen": 0}, {"gen": 1}]


@pytest.fixture
def matches():
    return [
        ({"gen": 0}, {"gen": 1}),
        ({"gen": 1}, {"gen": 0}),
        ({"gen": 1}, {"gen": 1}),
    ]


@pytest.fixture
def evals():
    return [[(1, -1)], [(1, -1)], [(-1, 0)]]


@pytest.fixture
def history():
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


@pytest.fixture
def model_tok(config):
    return get_model_tok("facebook/opt-125m", config)


def test_adapters(model_tok, config):
    model, _ = model_tok
    model.pretrained_model.load_adapter(config["peft"]["path"], adapter_name="A1")
    model.pretrained_model.set_adapter(adapter_name="A1")

    assert list(model.pretrained_model.peft_config.keys()) == ["default", "A1"]
    assert model.pretrained_model.active_adapter == "A1"

    model.pretrained_model.load_adapter(config["peft"]["path"], adapter_name="A2")
    model.pretrained_model.set_adapter(adapter_name="A2")

    assert list(model.pretrained_model.peft_config.keys()) == [
        "default",
        "A1",
        "A2",
    ]
    assert model.pretrained_model.active_adapter == "A2"

    model.pretrained_model.set_adapter(adapter_name="default")
    assert model.pretrained_model.active_adapter == "default"


def test_populate_with_entrant_adapters(model_tok, config):
    model, _ = model_tok

    league = []
    for generation in range(config["league"]["generations"]):
        entrants = league_entry(generation, config)
        league += entrants
        populate_with_entrant_adapters(model, entrants, config)

        # There's also the default adapter.
        assert len(league) == len(model.pretrained_model.peft_config) - 1

    # The adapter names should actually correspond to the player dicts.
    assert [
        json.loads(e)
        for e in model.pretrained_model.peft_config.keys()
        if e != "default"
    ] == league


def test_trajectories_by_model(league, matches, evals, history):
    sars = trajectories_by_model(league, matches, evals, history)

    assert list(sars[json.dumps(league[0])][0].values()) == [
        "Here's the board...",
        "A good opening would be...",
        1,
    ]


def test_update(model_tok, league, matches, evals, history, config):
    model, tok = model_tok
    populate_with_entrant_adapters(model, league, config)
    update(model, tok, league, matches, evals, history, config)


def test_abstract_train():
    league = train("facebook/opt-125m", play)

    assert isinstance(league, list)
    assert all([isinstance(e, dict) for e in league])
