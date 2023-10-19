from pelt.training import get_model_tok, populate_with_entrant_adapters, train
from pelt.defaults import default_config
from pelt.operators import league_entry
from pelt.games.tictactoe import play
import pytest
import pelt
import json


@pytest.fixture
def config():
    return default_config()


@pytest.fixture
def model_tok(config):
    return get_model_tok("facebook/opt-125m", config)


def test_adapters(model_tok, config):
    peft_model, _ = model_tok
    peft_model.load_adapter(config["peft"]["path"], adapter_name="A1")
    peft_model.set_adapter(adapter_name="A1")

    assert list(peft_model.peft_config.keys()) == ["default", "A1"]
    assert peft_model.active_adapter == "A1"

    peft_model.load_adapter(config["peft"]["path"], adapter_name="A2")
    peft_model.set_adapter(adapter_name="A2")

    assert list(peft_model.peft_config.keys()) == ["default", "A1", "A2"]
    assert peft_model.active_adapter == "A2"

    peft_model.set_adapter(adapter_name="default")
    assert peft_model.active_adapter == "default"


def test_populate_with_entrant_adapters(model_tok, config):
    peft_model, _ = model_tok

    league = []
    for generation in range(config["league"]["generations"]):
        entrants = league_entry(peft_model, generation, config)
        league += entrants
        populate_with_entrant_adapters(peft_model, entrants, config)

        # There's also the default adapter.
        assert len(league) == len(peft_model.peft_config) - 1

    # The adapter names should actually correspond to the player dicts.
    assert [
        json.loads(e) for e in peft_model.peft_config.keys() if e != "default"
    ] == league


def test_abstract_train():
    league = train("facebook/opt-125m", play)

    assert isinstance(league, list)
    assert all([isinstance(e, dict) for e in league])